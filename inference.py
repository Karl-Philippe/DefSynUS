import os
from PIL import Image
import numpy as np
import torch
from tqdm.auto import tqdm
import torchvision
import configargparse
from utils.plotter import Plotter
from utils.configargparse_arguments import build_configargparser
from utils.utils import argparse_summary
from cut.lotus_options import LOTUSOptions
import helpers
import trainer
from utils.classification_metrics import calculate_classification_metrics


if __name__ == "__main__":

    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser.add('--compute_metrics', action='store_true',
               help='compute per-label classification metrics (precision/recall) on the test set')
    parser, hparams = build_configargparser(parser)
    hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt_cut = LOTUSOptions().parse()   # get training options
    opt_cut.dataroot = hparams.data_dir_real_us_cut_training
    opt_cut.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')

    argparse_summary(hparams, parser)

    compute_all_metric = hparams.compute_metrics

    real_us_gt_test_dataloader, _ = helpers.load_real_us_gt_test_data(hparams)
    plotter = Plotter()    
    trainer = trainer.Trainer(hparams, opt_cut, plotter) 


    seg_network_ckpt = "./checkpoints/best_checkpoint_seg_renderer_valid_loss_33_exp_name25_5e-06_0.0001_0.0001_epoch=24.pt"
    cut_network_ckpt = "./checkpoints/best_checkpoint_CUT_val_loss_33_exp_name25_5e-06_0.0001_0.0001_epoch=24.pt"

    trainer.module.load_state_dict(torch.load(seg_network_ckpt))
    checkpoint = torch.load(cut_network_ckpt)
    # Create a new dictionary with keys without the "module." prefix
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    trainer.cut_trainer.cut_model.netG.load_state_dict(new_state_dict)

    
    # ------------------------------------------------------------------------------------------------------------------------------
    #             INFER REAL US IMGS THROUGH THE CUT+SEG NET - INFER THE WHOLE TEST SET
    # ------------------------------------------------------------------------------------------------------------------------------
    gt_test_imgs_plot_figs = []
    testset_losses = []

    if compute_all_metric:
        # Initialize variables to accumulate metrics
        accumulated_metrics = {label: {'TP': [], 'FP': [], 'FN': [], 'Precision': [], 'Recall': []} 
                            for label in range(1, 5)}  # assuming num_classes = 5, adjust accordingly
    
    with torch.no_grad():
        for nr, batch_data_real_us_test in tqdm(enumerate(real_us_gt_test_dataloader), total=len(real_us_gt_test_dataloader), ncols= 100, position=0, leave=True):
            
            real_us_test_img, real_us_test_img_label = batch_data_real_us_test[0].to(hparams.device), batch_data_real_us_test[1].to(hparams.device).float()
            reconstructed_us_testset = trainer.cut_trainer.cut_model.netG(real_us_test_img)

            reconstructed_us_testset = (reconstructed_us_testset / 2 ) + 0.5 # from [-1,1] to [0,1]

            testset_loss, seg_pred  = trainer.module.seg_forward(reconstructed_us_testset, real_us_test_img_label)
            print(f'testset_loss: {testset_loss}')
            testset_losses.append(testset_loss)

            if compute_all_metric:
                # Calculate classification metrics
                metrics = calculate_classification_metrics(seg_pred, real_us_test_img_label)

                # Accumulate metrics
                for label in range(1, 5):  # Adjust if num_classes is different
                    for metric in ['TP', 'FP', 'FN', 'Precision', 'Recall']:
                        accumulated_metrics[label][metric].append(metrics['mean_per_label'][label][metric])

            if hparams.logging and nr < hparams.nr_imgs_to_plot:
                real_us_test_img = (real_us_test_img / 2 ) + 0.5 # from [-1,1] to [0,1]

                plot_fig_gt = plotter.plot_stopp_crit(caption="testset_gt_|real_us|reconstructed_us|seg_pred|gt_label",
                                        imgs=[real_us_test_img, reconstructed_us_testset, seg_pred, real_us_test_img_label], 
                                        img_text='loss=' + "{:.4f}".format(testset_loss.item()), epoch='', plot_single=False, plot_val=True)
                gt_test_imgs_plot_figs.append(plot_fig_gt)

    if len(gt_test_imgs_plot_figs) > 0: 
        image_grid = torchvision.utils.make_grid(gt_test_imgs_plot_figs)

        # Convert tensor to image
        image_np = image_grid.permute(1, 2, 0).numpy()
        image_pil = Image.fromarray((image_np * 255).astype('uint8'))

        # Ensure the output folder exists
        output_folder = 'results_test'
        os.makedirs(output_folder, exist_ok=True)

        # Save the image locally
        image_pil.save(os.path.join(output_folder, "gt_test_imgs_plot_figs.png"))

    if compute_all_metric:
        # Calculate average and std for the accumulated metrics
        avg_metrics = {}
        std_metrics = {}
        for label in range(1, 5):  # Adjust if num_classes is different
            avg_metrics[label] = {metric: np.nanmean(accumulated_metrics[label][metric]) 
                                for metric in ['TP', 'FP', 'FN', 'Precision', 'Recall']}
            std_metrics[label] = {metric: np.nanstd(accumulated_metrics[label][metric]) 
                                for metric in ['TP', 'FP', 'FN', 'Precision', 'Recall']}
        
        # Print overall results
        print("\nOverall per Label Mean and Std:")
        for label in range(1, 5):  # Adjust if num_classes is different
            print(f"Label {label}:")
            for metric in ['TP', 'FP', 'FN', 'Precision', 'Recall']:
                print(f"  {metric}: {round(avg_metrics[label][metric], 3)} ± {round(std_metrics[label][metric], 3)}")

        # Initialize variables to calculate weighted precision and recall
        total_samples = 0
        weighted_precision_sum = 0
        weighted_recall_sum = 0
        weighted_precision_variance_sum = 0
        weighted_recall_variance_sum = 0

        # We assume the accumulated metrics are available per label
        for label in range(1, 5):  # Adjust if num_classes is different
            # Number of samples for the label (True Positives + False Negatives)
            TP = np.nansum(accumulated_metrics[label]['TP'])
            FN = np.nansum(accumulated_metrics[label]['FN'])
            num_samples = TP + FN
            total_samples += num_samples

            # Precision and recall values
            precision = np.nanmean(accumulated_metrics[label]["Precision"])
            recall = np.nanmean(accumulated_metrics[label]["Recall"])
            
            precision_std = np.nanstd(accumulated_metrics[label]["Precision"])
            recall_std = np.nanstd(accumulated_metrics[label]["Recall"])

            # Add weighted precision and recall
            weighted_precision_sum += precision * num_samples
            weighted_recall_sum += recall * num_samples

            # Variance for precision and recall (weighted)
            weighted_precision_variance_sum += (precision_std ** 2) * num_samples
            weighted_recall_variance_sum += (recall_std ** 2) * num_samples

        # Calculate final weighted averages
        weighted_precision = weighted_precision_sum / total_samples
        weighted_recall = weighted_recall_sum / total_samples

        # Calculate weighted standard deviation for precision and recall
        weighted_precision_std = np.sqrt(weighted_precision_variance_sum / total_samples)
        weighted_recall_std = np.sqrt(weighted_recall_variance_sum / total_samples)

        # Print the results
        print(f"Weighted Mean Precision: {weighted_precision:.3f} ± {weighted_precision_std:.3f}")
        print(f"Weighted Mean Recall: {weighted_recall:.3f} ± {weighted_recall_std:.3f}")

    
    avg_testset_loss = torch.mean(torch.stack(testset_losses))
    std_testset_loss = torch.std(torch.stack(testset_losses))
    print(f'\ntestset_gt_loss_epoch: {avg_testset_loss:.4f}')
    print(f'testset_gt_loss_std_epoch: {std_testset_loss:.4f}')
