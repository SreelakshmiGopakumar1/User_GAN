from Gray_No_Cond import train
import argparse, os
parser = argparse.ArgumentParser(description='GAN training parameters')
parser.add_argument('-d', dest='data_dir', help='dataset path')
parser.add_argument('-s', dest='save_dir', help='save model result')
parser.add_argument('-u', dest='user_eval', help='user feedback value')
parser.add_argument('-r', dest='restore_state', action='store_true', help='restore existing model')
parser.add_argument('-init', dest='restore_state', action='store_false', help='train from scratch')
parser.set_defaults(restore_state=False)
ARGS = parser.parse_args()

def main(data_dir, save_dir, user_eval, restore):
    save_dir = save_dir.replace('\\', '/')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    if restore:
        print('running after restore')
        path = save_dir.split('/')
        idx = int(path[-1])-1
        restore_dir = '/'.join(path[:-1]) + '/%d/checkpoint/model'%(idx)
        train(data_dir, save_dir, user_eval, restore_dir)
    else:
        print('running initial training')
        train(data_dir, save_dir, user_eval)

if __name__ == "__main__":
    main(ARGS.data_dir, ARGS.save_dir, float(ARGS.user_eval),\
         ARGS.restore_state)
