import utils
import os

run_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    config_alarm_path = f"{run_path}../docs/workshops/w2/data/play_tennis-train.csv"
    data = utils.formatIntoConfigStructureFile("A",config_alarm_path)
    print(data)
