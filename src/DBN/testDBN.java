/**
 * 
 */
package DBN;


import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import util.MNISTImageReader;
import util.Option;
import util.Util;

/**
 * @author peng
 *
 */
public class testDBN {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		MNISTImageReader r = new MNISTImageReader("train-images-idx3-ubyte");
		DoubleMatrix x = r.getDataMat();
		x = x.div(255);
		Option op = new Option();
		int[] size = {100};
		DBN dbn = new DBN(x, op, size);
		dbn.setup();
		dbn.train();
		dbn.rbm[0].b.print();
		dbn.rbm[0].c.print();

	}

}
