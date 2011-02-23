package sparse;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import sparse.Instance.Entry;




public class Dataset {
	ArrayList<Instance> data=new ArrayList<Instance>();
	
	int N, D, L;
	
	//take instances from d within [st,ed)
	Dataset(Dataset d,int st,int ed){
		this(d,st,ed,false);
	}	
	Dataset(Dataset d,int st,int ed,boolean reverse){
		N=d.N;
		L=d.L;
		if(!reverse){
			D=ed-st;
			for(int i=st;i<ed;i++)
				data.add(d.data.get(i));
		}else{
			D=d.D-(ed-st);
			for(int i=0;i<st;i++)
				data.add(d.data.get(i));
			for(int i=ed;i<d.D;i++)
				data.add(d.data.get(i));
		}
	}
	Dataset(File f) throws IOException{
		BufferedReader in=new BufferedReader(new FileReader(f));
		String[] ss = in.readLine().split(" ");
		D = Integer.valueOf(ss[0]); // # of instances
		N = Integer.valueOf(ss[1]); // # of features in each instance
		L = Integer.valueOf(ss[2]); // # if labels
		for (int i = 0; i < D; i++) {
			Instance ins=new Instance(N+1,L);//plus one bias variable
			ss = in.readLine().split(" ");
			ins.x.add(new Entry(0,1));
			for (int j = 0; j < N; j++)
				if(Double.valueOf(ss[j])!=0)
					ins.x.add(new Entry(j+1,Double.valueOf(ss[j])));

			for (int j = N; j < N + L; j++) {
				if(Double.valueOf(ss[j])==1.0){
					ins.y.add(j-N);
				}
			}
			data.add(ins);
		}
		N+=1;	//extra bias variable
		in.close();
	}
	Dataset[][] createCVData(int fold){
		Collections.shuffle(data);
		Dataset[][] res=new Dataset[fold][2];
		int VSetSize=D/fold;
		for(int i=0;i<fold;i++){
			res[i][0]=new Dataset(this,i*VSetSize,i*VSetSize+VSetSize,true);
			res[i][1]=new Dataset(this,i*VSetSize,i*VSetSize+VSetSize);
		}
		return res;
	}
	void setBias(double b){
		for(Instance ins:data)
			ins.x.get(0).val=b;
	}
}
