import numpy as np
import sys, math

class lbfgs(object):

    def __init__(self, m, tol=1e-4):
        self.m = m
        self.tol = tol

    def terminated(self, state):

        if not hasattr(self, 'vals'):
            self.vals = []

            return sys.maxsize

        self.vals.append(state.old_value)

        ret = abs((self.vals[-1] - self.vals[0]) / len(self.vals) / self.vals[-1]) if len(self.vals) > 2 else sys.maxsize
        if len(self.vals) > 5:
            self.vals.pop(0)
        return ret 
        return np.linalg.norm(state.new_grad)

    def minimize(self, differentiable_func, initial_x, reverse=False, iterations=100):
        state = lbfgs_direction(differentiable_func, initial_x, self.m, reverse=reverse)
        c = iterations

        while state.iter < c:

            print >> sys.stderr, 'Roud %d' % state.iter
            state.direction()
            state.backtracking_line_search()

            termination = self.terminated(state)
            print '%g' % termination
            if termination < self.tol:
                break
            state.update()

        return state.x

    def maximize(self, differentiable_func, initial_x):
        return self.minimize(differentiable_func, initial_x, reverse=True)
        

class lbfgs_direction(object):

    def __init__(self, differentiable_func, init, m, reverse=False):
        self.m = m
        self.f = differentiable_func
        self.s = []
        self.y = []
        self.alpha = []

        self.iter = 0
        self.reverse = -1.0 if reverse else 1.0

        self.x = init
        self.old_value, self.grad = self.f.value(self.x)

    def direction(self):

        # p is direction
        self.p = - self.grad
        n = len(self.s)
        i = n - 1

        if n <= 0:
            self.p = self.reverse * self.p
            return 

        while i >= 0:
            self.alpha[i] = np.dot(self.s[i], self.p) / np.dot(self.y[i], self.s[i])
            self.p = self.p - self.alpha[i] * self.y[i]
            i -= 1

        self.p = (np.dot(self.s[-1], self.y[-1]) / np.dot(self.y[-1], self.y[-1])) * self.p

        i = 0
        while i < n:
            beta = np.dot(self.y[i], self.p) / np.dot(self.s[i], self.y[i])
            self.p = self.p + (self.alpha[i] - beta) * self.s[i]
            i += 1
        self.p = self.reverse * self.p
        return 

    def backtracking_line_search(self):
        alpha = 1.0e-3
        backoff = 0.5

        if self.iter == 0:
            alpha = alpha / math.sqrt(np.dot(self.p, self.p))
            backoff = 0.1

        c1 = - 1.0e-4
#        print self.p, self.grad
        dir_grad = np.dot(self.p, self.grad)
        if dir_grad >= 0:
            print >> sys.stderr, 'non-descent direction: %f' % dir_grad
        while True:
            self.new_x = self.x + self.p * alpha
            print 'new_x, x, dir_grad', self.new_x, self.x, dir_grad, self.p, alpha
            value, self.new_grad = self.f.value(self.new_x)
            if self.reverse * self.old_value >= self.reverse * (value + c1 * dir_grad * alpha):
                self.old_value = value
                break

            if alpha <= 1.0e-50:
                raise

            alpha *= backoff
            print >> sys.stderr, 'Backtracking, %f, %f' % (alpha, backoff)

    def update(self):
        if len(self.s) > self.m:
            self.s.pop(0)
            self.y.pop(0)
            self.alpha.pop(0)

        self.s.append(self.new_x - self.x)
        self.y.append(self.new_grad - self.grad)
        self.alpha.append(0)

        self.x = self.new_x
        self.grad = self.new_grad
        self.iter += 1
