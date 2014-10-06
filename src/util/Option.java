package util;

/**
 * Store some info about train option.
 * @param numepochs	int the number of iteration times
 * @param	batchsize	int the size of mini-batch item
 * @param	momentum	double
 * @param	alpha	double	learning rate
 * @author peng
 *
 */
public class Option {
	public int numepochs = 1;
	public int batchsize = 100;
	public double momentum = 0;
	public double alpha = 1;
	public int push = 5;
	public int fetch = 10;
}
