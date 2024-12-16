import subprocess
import yaml

def update_dataDVC(data_ver):
   subprocess.run(["dvc", "add", "data"])
   subprocess.run(["git", "add", "data.dvc"])
   subprocess.run(["git", "commit", "-m", "create new dataset version"])
   tag = data_ver
   subprocess.run(["git", "tag", "-a", tag, "-m", "Dataset version"])
   subprocess.run(["git", "push"])
   subprocess.run(["dvc", "push"])

if __name__ == "__main__":

    print("Updating data version ...")

    with open('config.yaml') as f:
       params = yaml.safe_load(f)

    data_ver = params['Data_version']
    update_dataDVC(data_ver)
