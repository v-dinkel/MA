<?php


function getConnected($host,$user,$pass,$db) {

   $mysqli = new mysqli($host, $user, $pass, $db);

   if($mysqli->connect_error) 
     die('Connect Error (' . mysqli_connect_errno() . ') '. mysqli_connect_error());

   return $mysqli;
}

$mysqli = getConnected('localhost','root','','vesselanalysis');
$query = 'UPDATE results SET vesselClass = "'.$_GET['value'].'" WHERE id = '.$_GET['id'].';';

$data = $mysqli->query($query);
if (!$mysqli->commit()) {
    echo 'transaction failed';
    exit();
}else echo 'success';
?>