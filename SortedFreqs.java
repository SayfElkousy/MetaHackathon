import java.util.*;
public class SortedFreqs {
    public int[] freqs(String[] data) {
        // fill in code here
        TreeMap<String, Integer> storing = new TreeMap<>();
        for(int i = 0; i < data.length; i++){
            storing.put(data[i],storing.getOrDefault(data[i],0)+1);
        }
        List<Integer> numbers = new ArrayList<>(storing.values());
        int[] result = numbers.toArray(new int[0]);
      }
}
