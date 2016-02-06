def cal(x, t)
  x[0]*t[0] + x[1]*t[1] + x[2]*t[2] + x[3]*t[3]
end

t = [0,1,2,3]
t2 = [2,2,2,2] 

1.upto(40) do |n|
   x = [1,n,n+1,n+2]
  res1 = cal(x,t)
  res2 = cal(x,t2)
  puts "#{x.join(",")}  \t#{res1} \t#{res2}\t #{res1-res2}"
end

2.times do
  x = [1,rand(40),rand(40),rand(40)] 
  res1 = cal(x,t)
  res2 = cal(x,t2)
  puts "#{x.join(",")}  \t#{res1} \t#{res2}\t #{res1-res2}"
end
