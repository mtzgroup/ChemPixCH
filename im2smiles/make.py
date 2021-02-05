import glob

files = glob.glob("configs/data*.json")

names = [f.split("data_")[1] for f in files] # get names
names = [f.split(".")[0] for f in names] # rm .json
names = sorted(names)[::-1]
print names

instance = "train-{}:\n\tpython train.py --data=configs/data_{}.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/{}/\neval-{}:\npython evaluate_txt.py --results=results/{}/\n"

with open("makefile2", "w") as f:
    for name in names:
#        f.write(instance.format(name, name, name, name, name))
        f.write("train-{}:".format(name))
        f.write("\n")
        f.write("\tpython train.py --data=configs/data_{}.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/{}/".format(name, name))
        f.write("\n")
        f.write("eval-{}:".format(name))
        f.write("\n")
        f.write("\tpython evaluate_txt.py --results=results/{}/".format(name))
        f.write("\n\n")



'''
train-{}:
	python train.py --data=configs/data_{}.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/{}/

eval-{}:
	python evaluate_txt.py --results=results/{}/
'''
