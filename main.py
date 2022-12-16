import argparse
import re
from cpu import cpu
from gpu import gpu


def select_size(size_number, size_mul):
    size_number = int(size_number)
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
    if instrument == "CPU":
        if count > 1:
            empirical_mean_enc = 0
            empirical_mean_dec = 0
            for i in range(count):
                print(f'\n\tTest number #{i}\n')
                tmp1, tmp2 = cpu(size)
                if tmp1 != -1 and tmp2 != -1:
                    print("Successful!")
                    empirical_mean_enc += tmp1
                    empirical_mean_dec += tmp2
                else:
                    print("Oh no!")
            empirical_mean_enc = empirical_mean_enc / count
            empirical_mean_dec = empirical_mean_dec / count
            print(f'\nEmpirical_mean_enc = {round(empirical_mean_enc, 5)} Mb/sec\n'
                  f'Empirical_mean_dec = {round(empirical_mean_dec, 5)} Mb/sec')
        elif count == 1:
            if cpu(size):
                print("Successful!")
            else:
                print("Oh no! Error!")
    elif instrument == "GPU":
        if count > 1:
            empirical_mean_enc = 0
            empirical_mean_dec = 0
            for i in range(count):
                print(f'\n\tTest number #{i}\n')
                tmp1, tmp2 = gpu(size)
                if tmp1 != -1 and tmp2 != -1:
                    print("Successful!")
                    empirical_mean_enc += tmp1
                    empirical_mean_dec += tmp2
                else:
                    print("Oh no!")
            empirical_mean_enc = empirical_mean_enc / count
            empirical_mean_dec = empirical_mean_dec / count
            print(f'\nEmpirical_mean_enc = {round(empirical_mean_enc, 5)} Mb/sec\n'
                  f'Empirical_mean_dec = {round(empirical_mean_dec, 5)} Mb/sec')
        elif count == 1:
            if gpu(size):
                print("Successful!")
            else:
                print("Oh no! Error!")
    else:
        print("Error! Invalid syntax!")


if __name__ == '__main__':
    main()
