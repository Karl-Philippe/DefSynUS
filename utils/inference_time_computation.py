import torch
from tqdm.auto import tqdm
import configargparse
import time
from utils.configargparse_arguments import build_configargparser
from utils.utils import argparse_summary
from cut.lotus_options import LOTUSOptions
import helpers
import trainer

if __name__ == "__main__":
    # Configuration and device setup
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)
    hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt_cut = LOTUSOptions().parse()  # get training options
    opt_cut.dataroot = hparams.data_dir_real_us_cut_training
    opt_cut.device = hparams.device
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')

    argparse_summary(hparams, parser)

    real_us_gt_test_dataloader, _ = helpers.load_real_us_gt_test_data(hparams)
    trainer = trainer.Trainer(hparams, opt_cut, None)  # No plotter needed

    # Weights for different models
    weights = {
        # Unet + Lotus data augmentation
        "Unet_Lotus": {
            "seg_network_ckpt": "./checkpoints/best_checkpoint_seg_renderer_valid_loss_61_exp_name21_5e-06_0.0001_0.0001_epoch=74.pt",
            "cut_network_ckpt": "./checkpoints/best_checkpoint_CUT_val_loss_61_exp_name21_5e-06_0.0001_0.0001_epoch=74.pt"
        },
        # Unet + Advanced data augmentation
        "Unet_Advanced": {
            "seg_network_ckpt": "./checkpoints/best_checkpoint_seg_renderer_valid_loss_400_exp_name21_1e-06_0.0001_0.0001_epoch=13.pt",
            "cut_network_ckpt": "./checkpoints/best_checkpoint_CUT_val_loss_400_exp_name21_1e-06_0.0001_0.0001_epoch=13.pt"
        },
        # Attention Unet + Advanced data augmentation
        "Attention_Unet_Advanced": {
            "seg_network_ckpt": "./checkpoints/best_checkpoint_seg_renderer_valid_loss_378_exp_name21_5e-06_0.0001_0.0001_epoch=19.pt",
            "cut_network_ckpt": "./checkpoints/best_checkpoint_CUT_val_loss_378_exp_name21_5e-06_0.0001_0.0001_epoch=19.pt"
        }
    }

    # Select the model to evaluate
    model = "Attention_Unet_Advanced"
    seg_network_ckpt = weights[model]["seg_network_ckpt"]
    cut_network_ckpt = weights[model]["cut_network_ckpt"]

    # Load segmentation network
    trainer.module.load_state_dict(torch.load(seg_network_ckpt))

    # Load CUT network
    checkpoint = torch.load(cut_network_ckpt)
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    trainer.cut_trainer.cut_model.netG.load_state_dict(new_state_dict)

    # Inference loop
    inference_times = []

    with torch.no_grad():
        for _, batch_data_real_us_test in tqdm(enumerate(real_us_gt_test_dataloader), total=len(real_us_gt_test_dataloader), ncols=100, position=0, leave=True):            
            real_us_test_img, real_us_test_img_label = batch_data_real_us_test[0].to(hparams.device), batch_data_real_us_test[1].to(hparams.device).float()

            # Measure inference time
            start_time = time.time()
            reconstructed_us_testset = trainer.cut_trainer.cut_model.netG(real_us_test_img)
            reconstructed_us_testset = (reconstructed_us_testset / 2) + 0.5  # from [-1,1] to [0,1]
            testset_loss, seg_pred  = trainer.module.seg_forward(reconstructed_us_testset, real_us_test_img_label)
            end_time = time.time()

            inference_times.append(end_time - start_time)

    # Compute mean and std of inference times
    mean_inference_time = torch.mean(torch.tensor(inference_times))
    std_inference_time = torch.std(torch.tensor(inference_times))

    print(f'Mean inference time per image: {mean_inference_time:.5f} seconds')
    print(f'Std of inference time per image: {std_inference_time:.5f} seconds')
