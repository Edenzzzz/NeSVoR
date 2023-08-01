"""nesvor entrypoints"""


import sys
import string
import logging
from .parsers import main_parser
import os

def main() -> None:

    parser, subparsers = main_parser()
    # print help if no args are provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stdout)
        return
    if len(sys.argv) == 2:
        if sys.argv[-1] in subparsers.choices:
            subparsers.choices[sys.argv[-1]].print_help(sys.stdout)
            return
    # parse args
    args = parser.parse_args()

    ############################
    # Newly added args for O-INR
    ############################
    if args.o_inr:
            model_name = "O_INR"
    else:
        model_name = "INR"

    # setup log folder
    splited = args.output_volume.split("/")
    dir, name = "/".join(splited[:-1]), splited[-1]
    suffix = f"_{args.notes}" if args.notes else ""
    name = name.strip(".nii.gz") + f"_{model_name}_{args.n_iter}_iters" + suffix
    dir = os.path.join(dir, name)
    os.makedirs(dir, exist_ok=True)

    # setup file names inside
    name +=  ".nii.gz" 
    args.log_dir = dir
    args.output_volume = os.path.join(dir, name)
    args.mem_log = os.path.join(dir, "memory_cost.txt")
    args.debug_log = os.path.join(dir, "debug.txt")
    if os.path.exists(args.mem_log):
        os.remove(args.mem_log)
    if os.path.exists(args.debug_log):
        os.remove(args.debug_log)
        
    if not args.o_inr:
        # save hash grid for O-INR training 
        args.save_hash = True
    else:
        args.save_hash = False
        
    args.output_log = os.path.join(dir, "console_log.txt")    

    run(args)

def run(args) -> None:
    import torch
    from . import commands
    from .. import utils

    # setup logger
    if args.debug:
        args.verbose = 2
    utils.setup_logger(args.output_log, args.verbose)
    # setup device
    if args.device >= 0:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")
        logging.warning(
            "NeSVoR is running in CPU mode. The performance will be suboptimal. Try to use a GPU instead."
        )
    # setup seed
    utils.set_seed(args.seed)

    # execute command
    command_class = "".join(string.capwords(w) for w in args.command.split("-"))
    getattr(commands, command_class)(args).main()


if __name__ == "__main__":
    main()
