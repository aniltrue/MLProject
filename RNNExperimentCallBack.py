from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
import pandas as pd
from RNNLayers.AbstractRNN import AbstractRNN, AbstractBiRNN
import os
import matplotlib.pyplot as plt
import time
from datetime import timedelta


class RNNExperimentCallBack(Callback):
    def __init__(self, model: Model, dataset: str, batch_size: int,
                 log_dir: str = "logs/", experiments_path: str = "experiments.csv", metric: str = "acc"):

        super().__init__()

        self.log_dir = log_dir
        self.experiments_path = experiments_path
        self.model = model
        self.metric = metric
        self.batch_size = batch_size

        self.experiments = pd.read_csv(experiments_path, index_col=0)

        experiment_name = model.name
        experiment_id = self.experiments.shape[0]
        layer_name = ""
        layer_size = 0
        layer_activation = ""

        for layer in self.model.layers:
            if isinstance(layer, AbstractRNN):
                layer_name = layer.name
                layer_size = layer.cell.units
                layer_activation = layer.cell.kernel_activation.__name__
                break
            elif isinstance(layer, AbstractBiRNN):
                layer_name = layer.name
                layer_size = layer.cell_f.units
                layer_activation = layer.cell_f.kernel_activation.__name__
                break

        if layer_size == 0:
            raise Exception("Unknown RNN")

        row = {"ID": experiment_id,
               "Dataset": dataset,
               "Experiment Name": experiment_name,
               "RNN": layer_name,
               "Units": layer_size,
               "Activation": layer_activation,
               "Metric": self.metric,
               "Batch Size": batch_size}

        self.experiments = self.experiments.append(row, ignore_index=True)
        self.experiments.to_csv(experiments_path)
        self.id = experiment_id
        self.name = experiment_name

        self.format = "%s_%s_%d_%d_%s_%d" % (experiment_name, dataset, experiment_id, layer_size, layer_activation, batch_size)

        self.logs = pd.DataFrame(columns=["epoch", "loss", self.metric, "val_loss", "val_%s" % self.metric, "time"])

        self.training_start_time = 0
        self.epoch_start_time = 0

    def on_train_begin(self, logs=None):
        self.logs.to_csv(os.path.join(self.log_dir, "%s.csv" % self.format))
        print("-" * 50)
        print("Experiment %d(%s) training begins." % (self.id, self.name))
        print("-" * 50)

        self.training_start_time = time.time()

    def on_train_end(self, logs=None):
        elapsed_time = timedelta(seconds=time.time() - self.training_start_time)
        self.model.save_weights(os.path.join(self.log_dir, "%s.h5" % self.format))

        epochs = self.logs.shape[0]

        self.experiments.loc[self.experiments["ID"] == self.id, ["train_loss"]] = logs["loss"]
        self.experiments.loc[self.experiments["ID"] == self.id, ["train_metric"]] = logs[self.metric]
        self.experiments.loc[self.experiments["ID"] == self.id, ["val_loss"]] = logs["val_loss"]
        self.experiments.loc[self.experiments["ID"] == self.id, ["val_metric"]] = logs["val_%s" % self.metric]
        self.experiments.loc[self.experiments["ID"] == self.id, ["Epoch"]] = epochs
        self.experiments.loc[self.experiments["ID"] == self.id, ["Total Time"]] = elapsed_time

        self.experiments.to_csv(self.experiments_path)

        fig, axs = plt.subplots(1, 2)

        axs[0].plot(list(range(1, epochs + 1)), self.logs["loss"].to_numpy(), c="r", label="train")
        axs[0].plot(list(range(1, epochs + 1)), self.logs["val_loss"].to_numpy(), c="b", label="test")
        axs[1].plot(list(range(1, epochs + 1)), self.logs[self.metric].to_numpy(), c="r", label="train")
        axs[1].plot(list(range(1, epochs + 1)), self.logs["val_%s" % self.metric].to_numpy(), c="b", label="test")

        axs[0].set_title("Loss for %s" % self.name)
        axs[0].legend()
        axs[0].set_xticks(list(range(1, epochs + 1)))

        axs[1].set_title("Metric (%s) for %s" % (self.metric, self.name))
        axs[1].legend()
        axs[1].set_xticks(list(range(1, epochs + 1)))

        fig.tight_layout()
        plt.tight_layout()

        plt.savefig(os.path.join(self.log_dir, "%s.png" % self.format))
        plt.show()

        print("-" * 50)
        print("Experiment %d(%s) is completed in %s." % (self.id, self.name, str(elapsed_time)))
        print("Training Loss:\t{:.3f}".format(logs["loss"]))
        print("Validation Loss:\t{:.3f}".format(logs["val_loss"]))

        if self.metric == "acc" or self.metric == "accuracy":
            print("Training Accuracy:\t{:.2f}%".format(logs[self.metric] * 100))
            print("Validation Accuracy:\t{:.2f}%".format(logs["val_%s" % self.metric] * 100))
        else:
            print("Training {}:\t{:.4f}%".format(self.metric, logs[self.metric]))
            print("Validation {}:\t{:.4f}%".format(self.metric, logs["val_%s" % self.metric]))

        print("-" * 50)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        row = {"epoch": int(epoch),
               "loss": logs["loss"],
               self.metric: logs[self.metric],
               "val_loss": logs["val_loss"],
               "val_%s" % self.metric: logs["val_%s" % self.metric],
               "time": time.time() - self.epoch_start_time}

        self.logs = self.logs.append(row, ignore_index=True)

        self.logs.to_csv(os.path.join(self.log_dir, "%s.csv" % self.format))
