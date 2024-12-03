import numpy as np
import pandas as pd
import tqdm
import os
import math

def estimate_nodes(equation, max_d):
    cnt = 0
    total_sum = 0
    while 1:
        cnt += 1
        sum = (cnt*cnt*equation[0]+ cnt*equation[1] + equation[2])
        if ((equation[0] >= 0 and cnt > max_d) or cnt > (2*max_d)):
            break
        if (sum < 0):
            continue
        total_sum += sum
    return (total_sum)

def main():
    main_path = "/data/preprocessing/gbfs_lmcut"
    """
    paths = [
        main_path+"/airport/task09.pddl.csv",
        main_path+"/airport/task16.pddl.csv",
        main_path+"/airport/task17.pddl.csv",
        main_path+"/airport/task18.pddl.csv",
        main_path+"/airport/task19.pddl.csv",
        main_path+"/airport/task24.pddl.csv",
        main_path+"/blocks/task16.pddl.csv",
        main_path+"/blocks/task17.pddl.csv",
        main_path+"/blocks/task18.pddl.csv",
        main_path+"/blocks/task19.pddl.csv",
        main_path+"/blocks/task20.pddl.csv",
        main_path+"/blocks/task21.pddl.csv",
        main_path+"/blocks/task22.pddl.csv",
        main_path+"/blocks/task23.pddl.csv",
        main_path+"/blocks/task24.pddl.csv",
        main_path+"/blocks/task25.pddl.csv",
        main_path+"/blocks/task26.pddl.csv",
        main_path+"/blocks/task27.pddl.csv",
        main_path+"/blocks/task28.pddl.csv",
        main_path+"/blocks/task29.pddl.csv",
        main_path+"/blocks/task30.pddl.csv",
        main_path+"/blocks/task31.pddl.csv",
        main_path+"/blocks/task32.pddl.csv",
        main_path+"/blocks/task33.pddl.csv",
        main_path+"/blocks/task34.pddl.csv",
        main_path+"/blocks/task35.pddl.csv",
        main_path+"/depot/task03.pddl.csv",
        main_path+"/depot/task04.pddl.csv",
        main_path+"/depot/task07.pddl.csv",
        main_path+"/depot/task10.pddl.csv",
        main_path+"/depot/task13.pddl.csv",
        main_path+"/depot/task17.pddl.csv",
        main_path+"/elevators/task02.pddl.csv",
        main_path+"/elevators/task03.pddl.csv",
        main_path+"/elevators/task04.pddl.csv",
        main_path+"/elevators/task05.pddl.csv",
        main_path+"/elevators/task06.pddl.csv",
        main_path+"/elevators/task07.pddl.csv",
        main_path+"/elevators/task08.pddl.csv",
        main_path+"/elevators/task09.pddl.csv",
        main_path+"/elevators/task10.pddl.csv",
        main_path+"/elevators/task11.pddl.csv",
        main_path+"/elevators/task12.pddl.csv",
        main_path+"/elevators/task13.pddl.csv",
        main_path+"/elevators/task14.pddl.csv",
        main_path+"/elevators/task15.pddl.csv",
        main_path+"/elevators/task16.pddl.csv",
        main_path+"/elevators/task17.pddl.csv",
        main_path+"/elevators/task18.pddl.csv",
        main_path+"/elevators/task19.pddl.csv",
        main_path+"/elevators/task20.pddl.csv",
        main_path+"/elevators/task21.pddl.csv",
        main_path+"/elevators/task22.pddl.csv",
        main_path+"/elevators/task23.pddl.csv",
        main_path+"/elevators/task24.pddl.csv",
        main_path+"/elevators/task25.pddl.csv",
        main_path+"/elevators/task26.pddl.csv",
        main_path+"/elevators/task27.pddl.csv",
        main_path+"/elevators/task29.pddl.csv",
        main_path+"/freecell/task06.pddl.csv",
        main_path+"/freecell/task07.pddl.csv",
        main_path+"/freecell/task08.pddl.csv",
        main_path+"/freecell/task09.pddl.csv",
        main_path+"/freecell/task10.pddl.csv",
        main_path+"/freecell/task11.pddl.csv",
        main_path+"/freecell/task12.pddl.csv",
        main_path+"/freecell/task13.pddl.csv",
        main_path+"/freecell/task14.pddl.csv",
        main_path+"/freecell/task15.pddl.csv",
        main_path+"/freecell/task16.pddl.csv",
        main_path+"/freecell/task17.pddl.csv",
        main_path+"/freecell/task18.pddl.csv",
        main_path+"/freecell/task20.pddl.csv",
        main_path+"/gripper/task02.pddl.csv",
        main_path+"/gripper/task03.pddl.csv",
        main_path+"/gripper/task04.pddl.csv",
        main_path+"/gripper/task05.pddl.csv",
        main_path+"/logistics/task07.pddl.csv",
        main_path+"/logistics/task11.pddl.csv",
        main_path+"/logistics/task12.pddl.csv",
        main_path+"/logistics/task13.pddl.csv",
        main_path+"/logistics/task14.pddl.csv",
        main_path+"/logistics/task15.pddl.csv",
        main_path+"/logistics/task16.pddl.csv",
        main_path+"/logistics/task17.pddl.csv",
        main_path+"/logistics/task18.pddl.csv",
        main_path+"/logistics/task19.pddl.csv",
        main_path+"/logistics/task21.pddl.csv",
        main_path+"/miconic/task06.pddl.csv",
        main_path+"/miconic/task07.pddl.csv",
        main_path+"/miconic/task08.pddl.csv",
        main_path+"/miconic/task09.pddl.csv",
        main_path+"/miconic/task10.pddl.csv",
        main_path+"/openstacks/task02.pddl.csv",
        main_path+"/openstacks/task03.pddl.csv",
        main_path+"/openstacks/task04.pddl.csv",
        main_path+"/openstacks/task05.pddl.csv",
        main_path+"/openstacks/task06.pddl.csv",
        main_path+"/openstacks/task07.pddl.csv",
        main_path+"/parcprinter/task07.pddl.csv",
        main_path+"/parcprinter/task08.pddl.csv",
        main_path+"/parcprinter/task09.pddl.csv",
        main_path+"/parcprinter/task10.pddl.csv",
        main_path+"/parcprinter/task14.pddl.csv",
        main_path+"/parcprinter/task23.pddl.csv",
        main_path+"/parcprinter/task24.pddl.csv",
        main_path+"/parcprinter/task25.pddl.csv",
        main_path+"/pegsol/task08.pddl.csv",
        main_path+"/pegsol/task10.pddl.csv",
        main_path+"/pegsol/task12.pddl.csv",
        main_path+"/pegsol/task13.pddl.csv",
        main_path+"/pegsol/task14.pddl.csv",
        main_path+"/pegsol/task15.pddl.csv",
        main_path+"/pegsol/task16.pddl.csv",
        main_path+"/pegsol/task17.pddl.csv",
        main_path+"/pegsol/task18.pddl.csv",
        main_path+"/pegsol/task19.pddl.csv",
        main_path+"/pegsol/task20.pddl.csv",
        main_path+"/pegsol/task21.pddl.csv",
        main_path+"/pegsol/task22.pddl.csv",
        main_path+"/pegsol/task23.pddl.csv",
        main_path+"/pegsol/task24.pddl.csv",
        main_path+"/pegsol/task25.pddl.csv",
        main_path+"/pegsol/task26.pddl.csv",
        main_path+"/pegsol/task27.pddl.csv",
        main_path+"/psr-small/task15.pddl.csv",
        main_path+"/psr-small/task19.pddl.csv",
        main_path+"/psr-small/task22.pddl.csv",
        main_path+"/psr-small/task25.pddl.csv",
        main_path+"/psr-small/task29.pddl.csv",
        main_path+"/psr-small/task31.pddl.csv",
        main_path+"/psr-small/task35.pddl.csv",
        main_path+"/psr-small/task36.pddl.csv",
        main_path+"/psr-small/task40.pddl.csv",
        main_path+"/psr-small/task44.pddl.csv",
        main_path+"/psr-small/task46.pddl.csv",
        main_path+"/psr-small/task47.pddl.csv",
        main_path+"/rovers/task05.pddl.csv",
        main_path+"/rovers/task07.pddl.csv",
        main_path+"/rovers/task08.pddl.csv",
        main_path+"/rovers/task12.pddl.csv",
        main_path+"/satellite/task03.pddl.csv",
        main_path+"/satellite/task04.pddl.csv",
        main_path+"/scanalyzer/task02.pddl.csv",
        main_path+"/scanalyzer/task06.pddl.csv",
        main_path+"/scanalyzer/task07.pddl.csv",
        main_path+"/scanalyzer/task27.pddl.csv",
        main_path+"/scanalyzer/task28.pddl.csv",
        main_path+"/sokoban/task04.pddl.csv",
        main_path+"/sokoban/task05.pddl.csv",
        main_path+"/sokoban/task07.pddl.csv",
        main_path+"/sokoban/task08.pddl.csv",
        main_path+"/sokoban/task09.pddl.csv",
        main_path+"/sokoban/task10.pddl.csv",
        main_path+"/sokoban/task11.pddl.csv",
        main_path+"/sokoban/task12.pddl.csv",
        main_path+"/sokoban/task13.pddl.csv",
        main_path+"/sokoban/task14.pddl.csv",
        main_path+"/sokoban/task16.pddl.csv",
        main_path+"/sokoban/task17.pddl.csv",
        main_path+"/sokoban/task18.pddl.csv",
        main_path+"/sokoban/task19.pddl.csv",
        main_path+"/sokoban/task20.pddl.csv",
        main_path+"/sokoban/task21.pddl.csv",
        main_path+"/sokoban/task22.pddl.csv",
        main_path+"/sokoban/task23.pddl.csv",
        main_path+"/sokoban/task26.pddl.csv",
        main_path+"/sokoban/task27.pddl.csv",
        main_path+"/tpp/task05.pddl.csv",
        main_path+"/tpp/task06.pddl.csv",
        main_path+"/tpp/task07.pddl.csv",
        main_path+"/tpp/task08.pddl.csv",
        main_path+"/transport/task03.pddl.csv",
        main_path+"/transport/task04.pddl.csv",
        main_path+"/transport/task05.pddl.csv",
        main_path+"/transport/task14.pddl.csv",
        main_path+"/transport/task15.pddl.csv",
        main_path+"/transport/task24.pddl.csv",
        main_path+"/transport/task25.pddl.csv",
        main_path+"/transport/task26.pddl.csv",
        main_path+"/woodworking/task02.pddl.csv",
        main_path+"/woodworking/task03.pddl.csv",
        main_path+"/woodworking/task04.pddl.csv",
        main_path+"/woodworking/task12.pddl.csv",
        main_path+"/woodworking/task13.pddl.csv",
        main_path+"/woodworking/task14.pddl.csv",
        main_path+"/woodworking/task23.pddl.csv",
        main_path+"/woodworking/task24.pddl.csv",
        main_path+"/zenotravel/task06.pddl.csv",
        main_path+"/zenotravel/task08.pddl.csv",
        main_path+"/zenotravel/task11.pddl.csv",
    ]
    """
    ready_data_path = "/data/ready_data/gbfs_lmcut"
    domains =[
        ("airport", 30),
        ("blocks", 35),
        ("depot", 22),
        ("elevators", 30),
        ("freecell", 20),
        ("gripper", 20),
        ("logistics", 28),
        ("miconic", 30),
        ("movie", 30),
        ("openstacks", 30),
        ("parcprinter", 30),
        ("pegsol", 30),
        ("psr-small", 50),
        ("rovers", 30),
        ("satellite", 20),
        ("scanalyzer", 30),
        ("sokoban", 30),
        ("tpp", 30),
        ("transport", 30),
        ("woodworking", 30),
        ("zenotravel", 20),
    ]
    

    paths = []
    for domain in domains:
        temp_domain = domain[0]
        num_instances = domain[1]
        for instance in range(num_instances):
            if (instance<9):
                str_instance = "0"+str(instance+1)
            else:
                str_instance = str(instance+1)
            paths.append((main_path+"/"+temp_domain+"/task"+str_instance+".pddl.csv", ready_data_path+"/"+temp_domain+"_"+str_instance+"_x_.pt"))

    for path_tuple in paths:
        path = path_tuple[0]
        if os.path.isfile(os.getcwd()+path_tuple[1]) == False:
            continue
        column = ["level_1#", "level_1_H", "node_max"]
        df = pd.read_csv(os.getcwd()+path, skip_blank_lines=True, header=0, usecols=column)
        df = df.to_numpy()
        counter = {}
        estimation = []
        y_true = []
        print(len(df))
        for i in tqdm.tqdm(range(len(df))):
            temp = df[i][1]
            if math.isnan(temp):
                continue
            if int(temp) in counter: 
                counter[int(temp)] += 1
            else:
                counter[int(temp)] = 1
            x = []
            y = []
            x.append(0)
            y.append(0)
            for j in counter:
                x.append(j)
                y.append(counter[j])
            if(len(x) < 2):
                continue
            equation = np.polyfit(x, y, deg=2)
            if(equation[0] == 0):
                print("matan")
            estimation.append((i+1)/(estimate_nodes(equation, max(counter)))*100)
            y_true.append(abs((df[i][0]/df[i][2]*100) - estimation[i]))
        x_df = pd.DataFrame(estimation)
        words = path.split("/")
        x_df.to_csv("DBP/"+words[-2]+"-"+words[-1])
        y_df = pd.DataFrame(y_true)
        y_df.to_csv("DBP/y_"+words[-2]+"-"+words[-1])
if __name__ == '__main__':
    main()


 
    
