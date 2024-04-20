# Road Segmentation for RoboTour 2024

This repository contains the code for training and testing a road segmentation model for the RoboTour 2024 competition.

## Installation and setup

The code is meant to be developed locally in a `Python 3.10` environment and then run on a remote server.
Before we delve into the local and remote setup, we have to sign up to the [Weights & Biases](https://wandb.ai/)
platform
to log the training and testing process. After signing up, log in to the platform via the terminal by running the
following
command:

```bash
wandb login
```

This process is described in the [official documentation](https://docs.wandb.ai/quickstart).

### Local setup

1. Clone the repository and navigate to the root directory of the project.
   ```bash
   git clone https://github.com/aleskucera/road-segmentation.git
   cd road-segmentation
   ```
2. Install PyTorch by following the instructions on the [official website](https://pytorch.org/get-started/locally/).
3. Install other requirements by running the following command:
   ```bash
   pip install -r requirements.txt
   ```

### Remote setup

Remote machine should use Slurm Workload Manager to manage jobs. The following steps are for setting up the remote

1. Clone the repository and navigate to the root directory of the project.
    ```bash
    git clone https://github.com/aleskucera/road-segmentation
    cd road-segmentation
    ```

> [!NOTE]  
> There is no need to install any requirements on the remote machine as the dependencies are loaded in the Slurm
> script. The script will load the modules required for the job. Check the `.env` file for the list of modules.

## Usage

## Local usage

The code is divided into two main parts: training and testing. To run training on the local machine, run the following
command:

   ```bash
   python main.py action=train
   ```

To run testing on the local machine, run the following command:

   ```bash
   python main.py action=test ckpt_path=[checkpoint path]
   ```

You can also override the default configuration like `dataset` by passing the desired configuration as a command line
argument. For example, to change the dataset to `RoboTour`, run the following command:

   ```bash
   python main.py ds=robotour
   ```

For more details about default configuration and override options, run the following command:

   ```bash
   python main.py --help
   ```

## Remote usage

To run training on the remote machine, run the following command:

   ```bash
   sbatch [sbatch options] scripts/train.batch [script options]
   ```

To run testing on the remote machine, run the following command:

   ```bash
    sbatch [sbatch options] scripts/test.batch [script options]
   ```

For more details on the options, run the following command:

   ```bash
   bash scripts/train.batch --help
   ```

or

   ```bash
   bash scripts/test.batch --help
   ```