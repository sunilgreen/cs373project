import cart
import kfoldcv


def run():

    X, y = cart.load_data()
    print(kfoldcv.run(4, X, y))

run()
