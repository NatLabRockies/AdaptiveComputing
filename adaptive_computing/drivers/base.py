
class ActiveLoopDriverSF():
    def __init__(self, dataset, sampler, surrogate, eval_func):
        self.dataset = dataset
        self.sampler = sampler
        self.surrogate = surrogate
        self.eval_func = eval_func

    def step(self):
        self.surrogate.train(self.dataset.x_data,
                             self.dataset.y_data)

        x = self.sampler.get_sample(self.surrogate, self.dataset)
        y = self.evaluate_sample(x)

        self.dataset.add_samples(x,y)


    def run(self, N_steps):
        for i in range(N_steps):
            self.step()
        