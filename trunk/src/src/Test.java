import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;

import edu.stanford.nlp.optimization.CGMinimizer;
import edu.stanford.nlp.optimization.DiffFunction;


public class Test {
	static class F implements DiffFunction{

		@Override
		public double[] derivativeAt(double[] x) {
			// TODO Auto-generated method stub
			double[] res={3*x[0]*x[0]+2*x[0]-1};
			return res;
		}

		@Override
		public int domainDimension() {
			// TODO Auto-generated method stub
			return 1;
		}

		@Override
		public double valueAt(double[] x) {
			// TODO Auto-generated method stub
			return -(x[0]*x[0]*x[0]+x[0]*x[0]-x[0]+1);
		}
		
	}
	public static void main(String[] args){
		CGMinimizer opt=new CGMinimizer();
		double[] ini={0.0},res={-1.0};
		F f=new F();
		System.out.println(f.valueAt(opt.minimize(f, 1e-2, ini))+" "+f.valueAt(res));
	}
}
