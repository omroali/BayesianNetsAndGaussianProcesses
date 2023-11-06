import utils
import os

run_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    data_path = f"{run_path}/../docs/workshops/w2/data/play_tennis-train.csv"
    evaluation_variable = 'PT'
    data = utils.formatIntoConfigStructureFile("PT",data_path)
    print(data)
