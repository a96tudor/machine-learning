import numpy


class BinaryClassificationPerceptron:
    def __init__(self):
        self._lr = 0.0
        self._gamma = 0.0

        self.bias = 0

        self.weights = None

        self.xdim = None
        self.ydim = None

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, new_lr: float):
        if new_lr < 0 or new_lr > 1:
            raise ValueError(f'Learning rate should in the [0, 1] interval. Got {new_lr}')

        self._lr = new_lr

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, new_gamma: float):
        if not isinstance(new_gamma, float):
            raise ValueError(f'Gamma is expected to be a float. Got {type(new_gamma)}')
        self._gamma = new_gamma

    def train(self, x: numpy.ndarray, y: numpy.ndarray):
        if self.lr is None or self.gamma is None:
            raise ValueError('You need to set the learning rate and gamma value before training your model')

        self.__validate_shapes(x, y)
        self.weights = numpy.zeros(x.shape[1])
        err = self.gamma + 1
        step = 0

        while err > self.gamma:
            results = self.__run_training(x, y)
            err = self.__get_error(results, y)
            step += 1

            print(f'[STEP {step}] Error={err}, Error Delta={err-self.gamma}')

    def __validate_shapes(self, x: numpy.ndarray, y: numpy.ndarray):
        if len(x.shape) > 2 or len(x.shape) < 1:
            raise ValueError(f'Expected x to be a 2-dimensional array, got shape {x.shape}')
        if len(y.shape) > 1 or len(y.shape) == 0:
            raise ValueError(f'Expected y to be a 1-dimensional array, got shape {y.shape}')

        if x.shape[0] != y.shape[0]:
            raise ValueError(f'Expected x and y to have the same number of rows. Got {x.shape[0]} and {y.shape[0]}')

    def __run_training(self, x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        results = []
        for xi, true_yi in zip(x, y):
            yi = self.predict(xi)
            self.__update_weights(xi, yi, true_yi)
            results.append(yi)

        return numpy.array(results)

    def __get_error(self, y: numpy.ndarray, true_y: numpy.ndarray):
        s = len(y)

        return (1 / s) * sum(abs(yi - true_yi) for yi, true_yi in zip(y, true_y))

    def __update_weights(self, x: numpy.ndarray, y: int, true_y: int):
        new_weights = numpy.array([
            self.weights[i] + self.lr * (true_y - y) * x[i]
            for i in range(len(x))
        ])

        self.weights = new_weights

    def feed_forward(self, x: numpy.ndarray) -> float:
        return numpy.dot(self.weights, x) + self.bias

    def predict(self, x: numpy.ndarray) -> int:
        if self.feed_forward(x) + self.bias > 0:
            return 1
        else:
            return 0
