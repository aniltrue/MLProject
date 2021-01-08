from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
import pandas as pd
from RNNLayers.AbstractRNN import AbstractRNN, AbstractBiRNN
import os
import matplotlib.pyplot as plt


class RNNExperimentCallBack(Callback):
    def __init__(self, model: Model, dataset: str, metric: str = "acc",
                 log_dir: str = "logs/", experiments_path: str = "experiments.csv"):

        super().__init__()

        self.log_dir = log_dir
        self.experiments_path = experiments_path
        self.model = model
        self.metric = metric

        self.experiments = pd.read_csv(experiments_path)

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
               "Activation": layer_activation}

        self.experiments = self.experiments.append(row, ignore_index=True)
        self.experiments.to_csv(experiments_path)
        self.id = experiment_id
        self.name = experiment_name

        self.format = "%s_%s_%d_%d_%s" % (experiment_name, dataset, experiment_id, layer_size, layer_activation)

        self.logs = pd.DataFrame(columns=["epoch", "loss", self.metric, "val_loss", "val_%s" % self.metric])

    def on_train_begin(self, logs=None):
        self.logs.to_csv(os.path.join(self.log_dir, "%s.csv" % self.format))
        print("-" * 50)
        print("Experiment %d(%s) training begins." % (self.id, self.name))
        print("-" * 50)

    def on_train_end(self, logs=None):
        self.model.save_weights(os.path.join(self.log_dir, "%s.h5" % self.format))

        epochs = self.logs.shape[0]

        self.experiments.loc[self.experiments["ID"] == self.id, ["loss"]] = logs["loss"]
        self.experiments.loc[self.experiments["ID"] == self.id, ["metric"]] = logs[self.metric]
        self.experiments.loc[self.experiments["ID"] == self.id, ["val_loss"]] = logs["val_loss"]
        self.experiments.loc[self.experiments["ID"] == self.id, ["val_metric"]] = logs["val_%s" % self.metric]
        self.experiments.loc[self.experiments["ID"] == self.id, ["Epoch"]] = epochs

        self.experiments.to_csv(self.experiments_path)

        fig, axs = plt.subplots(1, 2)

        axs[0].plot(list(range(1, epochs + 1)), self.logs["loss"].to_numpy(), c="r", label="train")
        axs[0].plot(list(range(1, epochs + 1)), self.logs["val_loss"].to_numpy(), c="b", label="test")
        axs[1].plot(list(range(1, epochs + 1)), self.logs["metric"].to_numpy(), c="r", label="train")
        axs[1].plot(list(range(1, epochs + 1)), self.logs["val_metric"].to_numpy(), c="b", label="test")

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
        print("Experiment %d(%s) is completed." % (self.id, self.name))
        print("Training Loss:\t{:.3f}".format(logs["loss"]))
        print("Validation Loss:\t{:.3f}".format(logs["val_loss"]))

        if self.metric == "acc" or self.metric == "accuracy":
            print("Training Accuracy:\t{:.2f}%".format(logs[self.metric] * 100))
            print("Validation Accuracy:\t{:.2f}%".format(logs["val_%s" % self.metric] * 100))
        else:
            print("Training {}:\t{:.4f}%".format(self.metric, logs[self.metric]))
            print("Validation {}:\t{:.4f}%".format(self.metric, logs["val_%s" % self.metric]))

        print("-" * 50)

    def on_test_begin(self, logs=None):
        print("-" * 50)
        print("Experiment %d(%s) test begins." % (self.id, self.name))
        print("-" * 50)

    def on_test_end(self, logs=None):
        self.experiments.loc[self.experiments["ID"] == self.id, ["test_loss"]] = logs["loss"]
        self.experiments.loc[self.experiments["ID"] == self.id, ["test_metric"]] = logs[self.metric]

        self.experiments.to_csv(self.experiments_path)

        print("-" * 50)
        print("Experiment %d(%s) test is completed." % (self.id, self.name))
        print("Test Loss:\t{:.3f}".format(logs["loss"]))
        if self.metric == "acc" or self.metric == "accuracy":
            print("Test Accuracy:\t{:.2f}".format(logs[self.metric] * 100))
        else:
            print("Test {}:\t{:.4f}".format(self.metric, logs[self.metric]))

        print("-" * 50)

    def on_epoch_end(self, epoch, logs=None):
        row = {"epoch": epoch,
               "loss": logs["loss"],
               self.metric: logs[self.metric],
               "val_loss": logs["val_loss"],
               "val_%s" % self.metric: logs["val_%s" % self.metric]}

        self.logs = self.logs.append(row, ignore_index=True)

        self.logs.to_csv(os.path.join(self.log_dir, "%s.csv" % self.format))
