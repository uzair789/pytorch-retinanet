pid_list=$(screen -ls | awk '{print $1}')
for l in $pid_list; do
	sep=$(echo $l | tr "." "\n")
	p=$(echo $sep | awk '{print $1}')
	kill $p
done
