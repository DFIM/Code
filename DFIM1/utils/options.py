import argparse


def get_args():
    parser = argparse.ArgumentParser(description="TIRRS Args")
    ######################## general settings ########################
    parser.add_argument("--check_model", default='G', help="[O,R,C_16,G,Gmin]")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="TIRRS", help="experiment name to save")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", default=10)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/32') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.07, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=True, action='store_true')
    parser.add_argument("--txt_aug", default=False, action='store_true')
    parser.add_argument("--aug_ss", default=False, action='store_true')

    ## cross modal transfomer setting
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--MLM", default=True, action='store_true', help="whether to use Mask Language Modeling dataset")

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='itc', help="which loss to use ['mlm', 'dl', 'cyclic', 'itc', 'sdm', 'ot']")
    parser.add_argument("--mim_loss_weight", type=float, default=1, help="mim loss weight")
    parser.add_argument("--mlm_loss_weight", type=float, default=1, help="mlm loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    
    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(224, 224))
    parser.add_argument("--stride_size", type=int, default=32)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, AdamW]")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="RSICD", help="[RSITMD,RSICD]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    parser.add_argument("--num_instance", type=int, default=2)
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    parser.add_argument('--citc_lambda1', type=float, default=1)
    parser.add_argument('--citc_lambda2', type=float, default=1)
    parser.add_argument('--citc_gamma', type=float, default=1)
    parser.add_argument('--ot_gamma', type=float, default=0.5)
    parser.add_argument('--dl_gamma', type=float, default=0.1)
    parser.add_argument('--ss_loss_gamma', type=float, default=0.4)
	
    parser.add_argument('--seed', type=int, default=1356094784)

    args = parser.parse_args()

    return args