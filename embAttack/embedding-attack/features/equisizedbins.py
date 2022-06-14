import pandas as pd
import my_utils as utils

# test
import sys
import numpy
#numpy.set_printoptions(threshold=sys.maxsize)


def _create_bins(num_of_bins: int, diff: pd.DataFrame) -> [float]:
    labels = diff.index.values.tolist()

    # get list of all differences
    all_diff = []
    # create buckets
    for i in range(len(labels)):
        for j in range(i):
            all_diff.append(utils.get_difference(labels[i], labels[j], diff))

    print("all_diff:",len(all_diff),(len([x for x in all_diff if x==0])))
    print("#bins:",num_of_bins)
    # create bins
    #print("all diffs is:")
    #npdiffs = numpy.array(all_diff)
    #print(all_diff)

    _, bins = pd.qcut(all_diff, num_of_bins, retbins=True,duplicates='drop')
    bins = list(bins)

    return bins


class EquisizedBins:
    """
    Class for creating bins from diff inputs and
    the functionality to get the bin for a value
    """

    def __init__(self, num_of_bins: int, diff: pd.DataFrame):
        self.bins = _create_bins(num_of_bins, diff)

    def get_category(self, value) -> int:
        if self.bins[0] <= value <= self.bins[1]:
            return 0
        for i in range(1, len(self.bins) - 1):
            if self.bins[i] < value <= self.bins[i + 1]:
                return i
        raise ValueError(
            "Value does not fit in a bin. This should not happen since bins are created so that every value fits. \
            Value: {}\n Bins: {} ".format(
                value, self.bins))

    def get_number_of_bins(self) -> int:
        return len(self.bins) - 1
