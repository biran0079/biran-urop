package sparse;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class Instance {
	static class Entry{
		int idx;
		double val;
		Entry(int idx,double val){
			this.idx=idx;
			this.val=val;
		}
	};
	ArrayList<Entry> x;						//sparse feature
	Set<Integer> y;							//label set
	int N;									//length of sparse vector
	int L;									//number of distinct labels
	Instance(int N,int L){
		this.x=new ArrayList<Entry>();
		this.y=new HashSet<Integer>();
		this.N=N;
		this.L=L;
	}
	void printLabel(){
		System.out.print("[ ");
		for(Integer i:y)
			System.out.printf("%d ",i);
		System.out.print("]\n");
	}
	double dotProdX(double[] w,int base){
		double res=0;
		for (Entry e : x)
			res+=e.val*w[base+e.idx];
		return res;
	}
}
