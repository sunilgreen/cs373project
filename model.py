import c45_tree
import cart
import kfoldcv


def run():

    X, y = cart.load_data()
    print(kfoldcv.run(4, X, y, c45_tree, "c45", threshold=1.0))
    print(kfoldcv.run(4, X, y, cart, "cart", threshold=0.4))

run()
