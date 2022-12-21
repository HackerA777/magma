import argparse
import re
from cpu import cpu
from gpu import gpu


def select_size(size_number, size_mul):
    size_number = int(size_number)
    k = 0
    flag = True
    while flag:
        if 2 ** k == size_number:
            flag = False
        elif 2 ** k > size_number:
            return -1
        else:
            k += 1
    if size_mul == "mb":
        return size_number * 1024**2
    if size_mul == "gb":
        return 1024 ** 3
    else:
        return 1024 ** 2


def main():
    parser = argparse.ArgumentParser(description='main.py')
    size = ""
    instrument = ""

    parser.add_argument('-s', type=str, help="Объём данных, которые нужно зашифровать. nMb\\nGb, где n = 2^k. "
                                             "min = 1Mb; max = 1Gb",
                        required=True, dest='size')
    parser.add_argument('-i', type=str, help="Инструмент, с помощью, которого производится расчёт. CPU или GPU",
                        required=True,
                        dest='instrument')
    parser.add_argument('-c', type=int,
                        help="Количество повторений. После всех шифрований выводится "
                             "эмперическая средняя скорость шифрования",
                        required=False, dest='count', default=1)

    args = parser.parse_args()
    size = args.size
    instrument = args.instrument.upper()
    size_ = re.findall("\d+", size)
    size_mul = re.findall("\D+", size)
    size = select_size(size_[0], size_mul[0].lower())
    count = args.count
    if size == -1:
        print("Invalid size!")
        return
    if instrument == "CPU":
        if count > 1:
            empirical_mean_speed_enc = 0
            empirical_mean_speed_dec = 0
            empirical_mean_time_enc = 0
            empirical_mean_time_dec = 0
            for i in range(count):
                print(f'\n\tTest number #{i}\n')
                tmp1, tmp2, tmp3, tmp4 = cpu(size)
                if tmp1 != -1 and tmp2 != -1:
                    print("Successful!")
                    empirical_mean_speed_enc += tmp1
                    empirical_mean_speed_dec += tmp2
                    empirical_mean_time_enc += tmp3
                    empirical_mean_time_dec += tmp4
                else:
                    print("Oh no! Error!")
            empirical_mean_speed_enc = empirical_mean_speed_enc / count
            empirical_mean_speed_dec = empirical_mean_speed_dec / count
            empirical_mean_time_enc = empirical_mean_time_enc / count
            empirical_mean_time_dec = empirical_mean_time_dec / count
            print(f'\nEmpirical_mean_speed_enc = {round(empirical_mean_speed_enc, 5)} Mb/sec\n'
                  f'Empirical_mean_speed_dec = {round(empirical_mean_speed_dec, 5)} Mb/sec\n'
                  f'Empirical_mean_time_enc = {round(empirical_mean_time_enc, 5)} sec\n'
                  f'Empirical_mean_time_dec = {round(empirical_mean_time_dec, 5)} sec')
        elif count == 1:
            if cpu(size):
                print("Successful!")
            else:
                print("Oh no! Error!")
    elif instrument == "GPU":
        if count > 1:
            empirical_mean_speed_enc = 0
            empirical_mean_speed_dec = 0
            empirical_mean_all_time = 0
            for i in range(count):
                print(f'\n\tTest number #{i}\n')
                tmp1, tmp2, tmp3 = gpu(size)
                if tmp1 != -1 and tmp2 != -1:
                    print("Successful!")
                    empirical_mean_speed_enc += tmp1
                    empirical_mean_speed_dec += tmp2
                    empirical_mean_all_time += tmp3
                else:
                    print("Oh no! Error!")
            empirical_mean_speed_enc = empirical_mean_speed_enc / count
            empirical_mean_speed_dec = empirical_mean_speed_dec / count
            empirical_mean_all_time = empirical_mean_all_time / count
            print(f'\nEmpirical_mean_speed_enc = {round(empirical_mean_speed_enc, 5)} Mb/sec\n'
                  f'Empirical_mean_speed_dec = {round(empirical_mean_speed_dec, 5)} Mb/sec\n'
                  f'Empirical_mean_time_copying_and_encrypting = {round(empirical_mean_all_time, 5)} sec')
        elif count == 1:
            if gpu(size):
                print("Successful!")
            else:
                print("Oh no! Error!")
    else:
        print("Error! Invalid syntax!")


if __name__ == '__main__':
    main()
