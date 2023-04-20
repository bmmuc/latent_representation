import torch
# the dataset its [[x,y,vx,vy, id], ...]
# the possible values of id is: [0,1,2,3,4,5,6]
# x and y are in the range of [-1.2,1.2]
# vx and vy are in the range of [-1,1]
# an instance of the dataset must be in array of size 7 with unique values .

def generate_dataset():
    dataset = []
    for j in range(10000):
        instance = []
        possible_ids = list(range(0,7))
        for i in range(7):
            x = torch.rand(1)*2.4 - 1.2
            y = torch.rand(1)*2.4 - 1.2
            vx = torch.rand(1)*2 - 1
            vy = torch.rand(1)*2 - 1
            id = possible_ids[i] / 7
            instance.append(torch.FloatTensor([x,y,vx,vy,id]))

        dataset.append(instance)
    return dataset