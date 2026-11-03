import java.util.*;
public class SortByFreqs {
    public String[] sort(String[] data) {
        // fill in code here
        Map<String,Integer> map = new TreeMap<>();
        for(String s : data){
            map.put(s,map.getOrDefault(s,0)+1);
          }
        List<String> list = new ArrayList<>(map.keySet());
        String [] ret = list.toArray(new String[0]);
        Comparator<String> comp = new Comparator<String>(){
            public int compare(String a, String b){
                return mpa.get(b) - map.get(a);
            }
        };
        Arrays.sort(ret,comp);
        return ret;
      }
   }