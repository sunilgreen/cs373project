import cart
import kfoldcv


def run():

    X, y = cart.load_data()
    print(kfoldcv.run(3, X, y))

run()
