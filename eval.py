import numpy as np
import torch
from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (pprint, set_gpu,get_command_line_parser,postprocess_args)

import random
def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(1)
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    pprint(vars(args))

    set_gpu(args.gpu)
    trainer = FSLTrainer(args)
    
    trainer.evaluate_test()
    print(args.save_path)



