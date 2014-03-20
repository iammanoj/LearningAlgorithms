package cs246.homework.hw02;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.Map.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class kmeans extends Configured implements Tool {
   public static void main(String[] args) throws Exception {
      System.out.println(Arrays.toString(args));
      int res = ToolRunner.run(new Configuration(), new kmeans(), args);
      
      System.exit(res);
   }

   public int run(String[] args) throws Exception {
      System.out.println(Arrays.toString(args));
      
      
      String InitialCentroidDir = "hw02/centers";
      Path inputPath = new Path(args[0]);
      Path outputPath =null;
      Path centroidPath = null;  
      
      int NoOfIterations = 20;
      
      for (int iter = 0; iter < NoOfIterations;iter++){
    	  
    	  if (iter == 0){
    		  outputPath = new Path( args[1] +"_" + (iter+1));
    		  centroidPath =  new Path(InitialCentroidDir);
    	  }
    	  else if (iter == NoOfIterations -1){
    		  outputPath = new Path(args[1]);
    	  }
    	  else
    	  {
    		  centroidPath = outputPath;
    		  outputPath = new Path( args[1] +"_" + (iter+1));
    	  }
    	  
	      Job job = new Job(getConf(), "kmeans#"+iter);
	       
	      FileSystem fs = FileSystem.get(getConf());
	      
	      FileStatus[] fileStats = fs.listStatus(centroidPath);
	      System.out.println("Point#0: "+ fileStats.length);
	      
	      for (int i = 0; i < fileStats.length; i++) {
			   Path f = fileStats[i].getPath();
			  // System.out.println("Point#1: "+ f.toUri().toString());
			   if(f.toString().contains("part")){
				   DistributedCache.addCacheFile(f.toUri(),job.getConfiguration());
			   }
	      }
	      
	      job.setJarByClass(kmeans.class);
	      
	      job.setOutputKeyClass(Text.class);
	      job.setOutputValueClass(Text.class);
	
	      job.setMapperClass(Map.class);
	      job.setReducerClass(Reduce.class);
	
	      job.setInputFormatClass(TextInputFormat.class);
	      job.setOutputFormatClass(TextOutputFormat.class);
	
	      FileInputFormat.addInputPath(job, inputPath);
	      FileOutputFormat.setOutputPath(job, outputPath);
	
	      job.waitForCompletion(true);
      }
      return 0;
   }
   
   private static  double L2norm(ArrayList<Double> point, ArrayList<Double> centroid){
	   
	   if (point.size() == centroid.size()){
		   double norm = 0;
		   for(int i =0;i < point.size();i++){
			   norm += Math.pow((point.get(i) - centroid.get(i)),2);
		   }
		   return(Math.sqrt(norm));
	   }
	   else
	   {
		   System.err.println("ERROR : Vector Sizes are not same for centroid and point");
		   return 0;
	   }
   }
   
   private static  void String2Array(String str , ArrayList<Double> point){
	   
	   String[] S = str.split("[\\s\\n\\t]+");
	   for (String s :S){
		   point.add(Double.parseDouble(s));
	   }
   	}

   public static class Map extends Mapper<LongWritable, Text, Text, Text> {
  
	   ArrayList<ArrayList<Double>> centroids = new ArrayList<ArrayList<Double>> ();
	   
	   @Override
	   protected void setup(Context context)throws IOException,InterruptedException{
		
		//   System.out.println("Point #2 Inside Setup");
		   Configuration conf = context.getConfiguration();
		   FileSystem fs = FileSystem.get(conf);
		   Path[] cacheFiles = context.getLocalCacheFiles();
		   BufferedReader br = null;
		   
		//   System.out.println("Point# 2.5 "+ cacheFiles.length);
		   
		   for (int i = 0; i < cacheFiles.length; i++) {
		//	   System.out.println("Point#3 "+ cacheFiles[i].toUri().toString());
			
			   br = new BufferedReader( new FileReader( cacheFiles[i].toString()));
			   
			//   System.out.println("Point#3.2 ");
			   String line = "";
			   while ( (line = br.readLine() )!= null) {
				   ReadCentroid(line);
				//   System.out.println("point#4"+line);
				}
		   }
		   System.out.println("Point# 2.5 - size of intial K ="+ this.centroids.size());
	   }
	   private void ReadCentroid(String Line){
		   
		   //String[] arr = Line.split("[\\s\\n\\t]+");
		   ArrayList<Double> centroid = new ArrayList<Double>();
		   String2Array(Line , centroid);
		   this.centroids.add(centroid);
	   }
	   
	   private String NearestCentroid(ArrayList<Double> point){
		   
		   ArrayList<Double> ClosestCentroid = null;
		   
		   if (centroids.size() > 0){
			   
			   double L2_min = Double.MAX_VALUE;
			   for (ArrayList<Double> centroid : centroids ){
				   
				   double L2 = L2norm(point,centroid);
				   if (L2 <  L2_min) {
					   ClosestCentroid = centroid;
					   L2_min = L2;	   
				   }
			   }
			   StringBuilder c = new StringBuilder();
			   for(Double d : ClosestCentroid){
				   c.append(" "+d.toString());
				   
			   }
			   return(c.toString().trim());
		   }
		   else{
			   System.err.println("ERROR : no of centroids is zero");
			   return ("c");
		   }
	   
	   }
	   
      @Override
       public void map(LongWritable key, Text value, Context context)
              throws IOException, InterruptedException {
    	 
    	ArrayList<Double>  point = new ArrayList<Double>();
    	String2Array( value.toString() , point);
    	
     	String  nearest = NearestCentroid(point);
 		context.write(new Text(nearest), value);
 		
      }
   }

   public static class Reduce extends Reducer<Text, Text, NullWritable,Text > {
      @Override
      public void reduce(Text key, Iterable<Text> values, Context context)
              throws IOException, InterruptedException {
    	  
    	  String NewCentroid = CalcCentroid(key,values);
    	  
    	  context.write( NullWritable.get(), new Text(NewCentroid));
      }
      
      private String CalcCentroid(Text key, Iterable<Text> values){
    	  
    	 int count = 0;
    	 ArrayList<Double> NewCentroid = new ArrayList<Double>();
    	 ArrayList<Double> point  = null;
    	 ArrayList<Double> OldCentroid =new ArrayList<Double>();

    	 String2Array(key.toString() , OldCentroid);
    	 
    	 for(int i = 0; i < OldCentroid.size(); i++){
			 NewCentroid.add(0.0);
		 }
    	 double Cost = 0.0;
    	 
    	 for(Text p : values){    		 
    		 point = new  ArrayList<Double>();
    		 String2Array( p.toString() , point);
    		 
    		 Cost +=  Math.pow(L2norm(point, OldCentroid),2);
    		 for(int i = 0; i < point.size() ; i++){
    			 NewCentroid.set(i, NewCentroid.get(i) + point.get(i));
    		 }
    		 count++;
    	 }
    	 //System.out.println("From Reducer : Cost is "+Cost);
    	 System.out.println(String.format("%.2f",Cost));
    	 
    	 StringBuilder nc = new StringBuilder();
    	 for(int i = 0; i < NewCentroid.size(); i++){
			 double ncoor =  NewCentroid.get(i)/count;
			 nc.append(" "+ ncoor);
		 }
    	  return (nc.toString().trim());
      }
      
   }
}
