import argparse

from azureml.core import ScriptRunConfig
from azureml.core import Workspace
from azureml.core.experiment import Experiment

from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException

from azureml.core import Datastore
from azureml.core.runconfig import DataReferenceConfiguration

from azureml.core.runconfig import RunConfiguration, DEFAULT_GPU_IMAGE
from azureml.core.conda_dependencies import CondaDependencies

import configparser
import ast
import sys

def connect():
    workspace = Workspace.from_config(path="./config.json")
    print('Connected to Azure')

    return workspace


def run(workspace, config, args):
    compute_target_name = config['train']['compute_target_name']
    data_folder = config['train']['data_folder']

    try:
        compute_target = ComputeTarget(workspace=workspace, name=compute_target_name)
        print('found existing:', compute_target.name)
    except ComputeTargetException:
        print('creating new.')
        compute_config = AmlCompute.provisioning_configuration(
            vm_size=config['train']['vm_size'],
            min_nodes=0,
            max_nodes=1)
        compute_target = ComputeTarget.create(workspace, compute_target_name, compute_config)
        compute_target.wait_for_completion(show_output=True)

    # ds = Datastore.register_azure_blob_container(
    #     workspace,
    #     datastore_name=config['train']['datastore_name'],
    #     account_name=config['train']['account_name'],
    #     account_key=config['train']['account_key'],
    #     container_name=config['train']['container_name'],
    #     overwrite=True)
    #
    # # # Upload local "data" folder (incl. files) as "tfdata" folder
    # ds.upload(
    #     src_dir=config['train']['local_directory'],
    #     target_path=data_folder,
    #     overwrite=True)

    ds = Datastore.get(workspace, datastore_name=config['train']['datastore_name'])

    # generate data reference configuration
    dr_conf = DataReferenceConfiguration(
        datastore_name=ds.name,
        path_on_datastore=data_folder,
        mode='mount')  # set 'download' if you copy all files instead of mounting

    run_config = RunConfiguration(
        framework="python",
        conda_dependencies=CondaDependencies.create(conda_packages=ast.literal_eval(config['train']['conda_packages'])))
    run_config.target = compute_target.name
    run_config.data_references = {ds.name: dr_conf}
    run_config.environment.docker.enabled = True
    # run_config.environment.docker.gpu_support = True
    run_config.environment.docker.base_image = DEFAULT_GPU_IMAGE

    src = ScriptRunConfig(
        source_directory='./script',
        script='train.py',
        run_config=run_config,
        arguments=['--datadir', str(ds.as_mount()),
                   '--step', args.step,
                   '--train_on', args.train_on,
                   '--fold', args.fold,
                   '--epochs', args.epochs,
                   '--experiment', args.experiment,
                   '--reference', args.reference,
                   '--batchsize', args.batchsize,
                   '--optimizertype', args.optimizertype,
                   '--convrnn_filters', args.convrnn_filters,
                   '--learning_rate', args.learning_rate,
                   '--pix250m', args.pix250m]
    )
    # exp = Experiment(workspace=ws, name='test20181210-09')
    exp = Experiment(workspace=workspace, name=config['train']['experiment_name'])
    run = exp.submit(config=src)
    run.wait_for_completion(show_output=True)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--train_on', type=str, default="2002",
                        help='Dataset partition to train on. Datasets are defined as sections in dataset.ini in datadir')
    parser.add_argument('-b', '--batchsize', type=int, default=32,
                        help='batchsize')
    parser.add_argument('-f', '--fold', type=int, default=0,
                        help="fold (requires train<fold>.ids)")
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help="epochs")
    parser.add_argument('-step', '--step', type=str, default="training",
                        help='step')
    parser.add_argument('-experiment', '--experiment', type=str, default="bands",
                        help='Experiment to train')
    parser.add_argument('-ref', '--reference', type=str, default="MCD12Q1v6stable01to15_LCProp2_major",
                        help='Reference dataset to train')
    parser.add_argument('--learning_rate', type=float, default=0.08,
                        help="overwrite learning rate. Required placeholder named 'learning_rate' in model")
    parser.add_argument('--convrnn_filters', type=int, default=24,
                        help="number of convolutional filters in ConvLSTM/ConvGRU layer")
    parser.add_argument('-optimizertype', '--optimizertype', type=str, default="adam",
                        help='optimizertype')
    parser.add_argument('--pix250m', type=int, default=24,
                        help="pix250m")

    # args, _ = parser.parse_known_args()
    args, _ = parser.parse_known_args(args=argv[1:])
    return args

def main(args):
    # Read config file
    config = configparser.ConfigParser()
    config.read('ml-config.ini')

    workspace = connect()
    run(workspace, config, args)

if __name__ == '__main__':
    args = parse_arguments(sys.argv)
    main(args)