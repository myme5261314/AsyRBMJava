/**
 * 
 */
package util;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

import org.jblas.DoubleMatrix;

/**
 * @author peng
 *
 */
public class MNISTLabelReader {
	
	
	public String path = "../train-labels-idx1-ubyte";
	public static double[] data;
	public int num;
	public MNISTLabelReader(String path) {
		// TODO Auto-generated constructor stub
		this.path = path;
	}
	public double[] loadData() {
		try {
			DataInputStream in = new DataInputStream(new FileInputStream(this.path));
			int magic;
			magic = in.readInt();
			assert(magic==2049);
			num = in.readInt();
			byte[] buffer = new byte[num];
			data = new double[num];
			in.read(buffer);
			in.close();
			for(int i=0; i<num; i++) {
				data[i] = (double) (buffer[i] & 0xFF);
			}
			
		} catch (FileNotFoundException e) {
			// TODO: handle exception
		} catch (IOException e) {
			
		}
		return data;
	}
	public DoubleMatrix getDataMat() {
		double[] data = this.loadData();
//		return new DoubleMatrix(num, row*col, data);
		DoubleMatrix out = new DoubleMatrix(num, 1, data);
		return out;
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		MNISTLabelReader r = new MNISTLabelReader("train-labels-idx1-ubyte");
		DoubleMatrix mat = r.getDataMat();
		mat.print();
////		mat.print();
////		mat = mat.divide(255);
//		mat = mat.extractMatrix(1, 2, 0, DoubleMatrix.);
//		mat.print();
//		mat.reshape(r.row, r.col);
//		mat.print();
	}

}
