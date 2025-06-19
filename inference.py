import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
import tensorflow as tf
from keras.optimizers import Adam
import click, sys
from utils import *

def option(f):
    options = [
        click.option('--original_file', type=click.STRING, default='./data/original_data.txt'),
        click.option('--predict_input_file', type=click.STRING, default='./data/baseline_added_data_%d.txt'),
        click.option('--random_state', type=click.INT, default=42),
    ]
    for option in reversed(options):
        f = option(f)
    return f

def predict_parameter(file_name, trained_model, num):
    predict_input_data = []
    data_array = []
    with open(file_name % num, 'r') as file:
        for line in file:
            data_point = line.strip()
            data_array.append(float(data_point))
    predict_input_data.append([data_array])

    predict_param = trained_model.predict(predict_input_data, verbose=0)
    return predict_param

@click.command()
@option
def run(original_file, predict_input_file, random_state):
    set_seed(random_state)

    # Inference
    parameters = ['lambda_', 'itermax', 'dssn_criterion', 'porder']
    data_len = len(pd.read_csv(original_file, header=None).values)
    for param_name in parameters:
        trained_model = tf.keras.models.load_model(f'./model/{param_name}.h5', custom_objects={"RMSE": RMSE, "SMAPE": SMAPE})

        globals()[f'{param_name}_list'] = []
        for i in range(4):
            predicted = predict_parameter(predict_input_file, trained_model, i)
            predicted_rescaled = rescaling_parameter(predicted, param_name)
            if param_name == 'lambda_' or param_name == 'dssn_criterion':
                globals()[f'{param_name}_list'].append(predicted_rescaled)
            else:
                globals()[f'{param_name}_list'].append(predicted_rescaled[0][0])

    # Visualization
    for i in range(4):
        compare_graph_plot(data_len, predict_input_file % i, original_file, lambda__list[i], itermax_list[i], dssn_criterion_list[i], porder_list[i])
        plt.savefig(f'./compare-graph-{i}.png')
        plt.show(block=False)
        plt.clf()

if __name__ == '__main__':
    run()