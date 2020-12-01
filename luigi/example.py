import luigi
import sklearn.linear_model

from base import BaseAggregateMetrics
from base import BaseLearn
from base import copy_file
from params import my_params


class AggregateMetrics(BaseAggregateMetrics):
    output_dir = luigi.Parameter("tmp")

    @copy_file(__file__)
    def requires(self):
        for my_param in my_params:
            yield [Learn(output_dir=self.output_dir, **my_param)]


class Learn(BaseLearn):
    output_dir = luigi.Parameter()
    seed = luigi.IntParameter()
    data = luigi.DictParameter()
    model = luigi.DictParameter()

    # define your functions if you need
    # def load_data():
    # def train():


if __name__=='__main__':
    luigi.run(['AggregateMetrics', '--workers', '1', '--local-scheduler'])
