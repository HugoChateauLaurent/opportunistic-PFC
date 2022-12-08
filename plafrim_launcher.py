import pickle
import argparse
import opportunistic_pfc
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--codepath', required=True, help='path to code')
    parser.add_argument('--simpath', required=True, help='path to parameter file')
    args = parser.parse_args()

    params = pickle.load(open(args.simpath+"/params.pickle",'rb'))
    simpath = os.path.abspath(args.simpath)
    os.chdir(args.codepath)

    params["xp_cls"].simulate(simpath, params)
