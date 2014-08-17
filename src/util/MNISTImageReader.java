package util;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

import org.jblas.DoubleMatrix;

public class MNISTImageReader {
	public String path = "../train-images-idx3-ubyte";
	public static double[] data;
	public int row;
	public int col;
	public int num;
	public MNISTImageReader(String path) {
		// TODO Auto-generated constructor stub
		this.path = path;
	}
	public double[] loadData() {
		try {
			DataInputStream in = new DataInputStream(new FileInputStream(this.path));
			int magic;
			magic = in.readInt();
			assert(magic==2051);
			num = in.readInt();
			row = in.readInt();
			col = in.readInt();
			byte[] buffer = new byte[num*row*col];
			data = new double[num*row*col];
			in.read(buffer);
			in.close();
			for(int i=0; i<num*row*col; i++) {
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
		DoubleMatrix out = new DoubleMatrix(col*row, num, data);
		out = out.transpose();
		for (int i = 0; i < out.getRows(); i++) {
			DoubleMatrix r = out.getRow(i);
			out.putRow(i, r.reshape(col, row).transpose().reshape(1, row*col));
		}
		return out;
	}
	
	public static void main(String args[]) {
//		MNISTImageReader r = new MNISTImageReader("train-images-idx3-ubyte");
//		double[] d = r.loadData();
//		DoubleMatrix mat = new DoubleMatrix(r.num, r.row*r.col, d);
////		mat.print();
////		mat = mat.divide(255);
//		mat = mat.extractMatrix(1, 2, 0, DoubleMatrix.);
//		mat.print();
//		mat.reshape(r.row, r.col);
//		mat.print();
		
	}
}
