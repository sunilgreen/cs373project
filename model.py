import c45_tree
import cart
import kfoldcv


def run():

    X, y = cart.load_data()
    for i in [x / 2 for x in range(30)]:
        print("Threshold: "+str(i))
        print(kfoldcv.run(4, X, y, c45_tree, "c45", threshold=float(i)))
    #print(kfoldcv.run(4, X, y, cart, "cart", threshold=0.4))

run()
