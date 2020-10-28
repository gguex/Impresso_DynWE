import re

with open("/home/gguex/Gride_search_results.txt", "r") as file:
    str_to_p = file.readline()

split_str = re.split(";", str_to_p)

with open("/home/gguex/Gride_search_results_test.txt", "w") as file:
    for i in range(len(split_str)):
        if split_str[i] == "300":
            if split_str[i-1][-1] == "7":
                file.write("\n" + split_str[i-1][-1] + ";" + split_str[i])
            else:
                file.write("\n" + split_str[i-1][-2:] + ";" + split_str[i])
        else:
            file.write(";" + split_str[i])

