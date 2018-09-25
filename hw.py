#!/usr/bin/env python3

from abc import ABC, abstractmethod

import argparse
import math
import matplotlib.pyplot as plt
import random


class Distribution(ABC):
    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def predict_parameter(self, k, avg):
        pass

    def sample(self, size):
        return [self.experiment() for i in range(size)]

    def mean_moment(self, k, sample_size):
        return sum(map(lambda x: x ** k, self.sample(sample_size))) / sample_size

    def prediction(self, k, sample_size):
        return self.predict_parameter(k, self.mean_moment(k, sample_size))


class UniformDistribution(Distribution):
    def __init__(self, theta):
        self.__theta = theta

    def experiment(self):
        return random.uniform(0, self.__theta)

    def predict_parameter(self, k, avg):
        # E[X^k] = theta^k / (k + 1)
        return ((k + 1) * avg) ** (1 / k)


class ExponentialDistribution(Distribution):
    def __init__(self, theta):
        self.__theta = theta

    def experiment(self):
        # It seems from the task that we should create an exponential
        # distribution with mean value equal to theta
        return random.expovariate(1 / self.__theta)

    def predict_parameter(self, k, avg):
        # E[X^k] = k! / lambda^k = k! theta^k
        return (avg / math.factorial(k)) ** (1/k)


class Params:
    def __init__(self, theta, sample_size, exps):
        self.theta = theta
        self.sample_size = sample_size
        self.exps = exps


def dist_std_deviation(distclass, k, params):
    dist = distclass(params.theta)
    exps = params.exps
    values = [dist.prediction(k, params.sample_size) for i in range(exps)]
    deviations = map(lambda x: (x - params.theta) ** 2, values)
    return sum(deviations) / exps


def test_distribution(distclass, maxk, params):
    return [dist_std_deviation(distclass, k + 1, params) for k in range(maxk)]


DISTCLASS_MAP = {
    'uniform': UniformDistribution,
    'exponential': ExponentialDistribution
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Random seed", type=int)
    parser.add_argument("maxk", help="Maximal index of moment", type=int)
    parser.add_argument("theta", help="Value of parameter to approximate",
                        type=float)
    parser.add_argument("sample_size",
                        help="Size of samples for one experiment", type=int)
    parser.add_argument("exps", help="Number of experiments for each moment",
                        type=int)
    parser.add_argument("distribution", help="The distribution to use",
                        choices=DISTCLASS_MAP.keys())
    parser.add_argument("--output", help="The result file", default="plot.png")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    params = Params(args.theta, args.sample_size, args.exps)
    distclass = DISTCLASS_MAP[args.distribution]

    result = test_distribution(distclass, args.maxk, params)

    plt.xlabel('Moment')
    plt.ylabel('Standard deviation')
    plt.suptitle('{} distribution'.format(args.distribution))
    plt.plot(range(1, args.maxk + 1), result, '-')
    plt.savefig(args.output)


if __name__ == '__main__':
    main()
