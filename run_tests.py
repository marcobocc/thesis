import argparse
from Tester import Tester


parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-f", "--filename", type=str)
group.add_argument("-v", "--validation", type=int)

args = parser.parse_args()
filename = args.filename
validation_tests = args.validation

tester = Tester()
if (validation_tests):
    tester.run_autogenerated_tests(validation_tests)
elif (filename):
    tester.run_groundtruth_tests(filename)
