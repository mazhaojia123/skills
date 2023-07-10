import argparse
import os
import subprocess
import csv


def process_csv():
    # get csv file
    if path == None:
        log_tmp_path = '.log_tmp'
        log_tmp_csv_path = '.log_tmp.csv'
    else:
        log_tmp_path = path + '/' + '.log_tmp'
        log_tmp_csv_path = path + '/' + '.log_tmp.csv'

    with open(log_tmp_path) as in_file, \
            open(log_tmp_csv_path, 'w') as out_file:
        for aline in in_file:
            if not aline.startswith('=='):
                out_file.write(aline)

    # calculate the final result
    with open(log_tmp_csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        next(csv_reader)
        sm_effi, exec_time = [], []

        for row in csv_reader:
            if row[9] == 'gpu__time_active.avg':
                exec_time.append(int(row[11].replace(',', '')))
            if row[9] == 'smsp__cycles_active.avg.pct_of_peak_sustained_elapsed':
                sm_effi.append(float(row[11]))

    the_sum = sum(exec_time)
    avg = 0.0
    for i in range(len(sm_effi)):
        avg += sm_effi[i] * (exec_time[i]/the_sum)

    print("The overall sm_efficiency of " + args.filename + " is ", avg)


def ncu_exec():
    command = ['ncu', '--csv', '--log-file', '.log_tmp',
               '--metrics',
               'smsp__cycles_active.avg.pct_of_peak_sustained_elapsed' +
               ',' + 'gpu__time_active.avg',
               real_file_name]

    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         cwd=path)
    out, err = p.communicate()

    print(out.decode('UTF-8'))
    print(err.decode('UTF-8'))


if __name__ == "__main__":
    # process the cli argument
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="The name of the program which we are about to profile.", type=str)
    parser.add_argument("--disablencu", help="disable ncu so that only process the csv file.", action="store_true")
    args = parser.parse_args()

    path = os.path.dirname(args.filename)
    real_file_name = os.path.basename(args.filename)
    if len(path) == 0:
        path = None

    if args.disablencu == False:
        ncu_exec()

    process_csv()
