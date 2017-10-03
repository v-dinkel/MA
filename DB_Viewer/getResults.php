<?php


function getConnected($host,$user,$pass,$db) {

   $mysqli = new mysqli($host, $user, $pass, $db);

   if($mysqli->connect_error) 
     die('Connect Error (' . mysqli_connect_errno() . ') '. mysqli_connect_error());

   return $mysqli;
}

$mysqli = getConnected('localhost','root','','vesselanalysis');
$query = "";
if (strcmp($_GET['query'], 'no') !== 0){
	$query = $_GET['query'];
}else{
	$query = "SELECT * FROM results";
}

$data = $mysqli->query($query)->fetch_all(MYSQLI_ASSOC);
// getting keys from the first row
$header = array_keys(reset($data));
// printing them
$index = 0;
$verifiedIndex = 0;
$verifiedOccurs = False;
$vesselIndex = 0;
$vesselOccurs = False;
$idIndex = 0;
$idOccurs = False;
$indeces = array();
$zoneOccurs = False;
$zoneIndex = 0;
$sequenceIdOccurs = False;
$sequenceIdIndex = 0;
$vesselClassOccurs = False;
$vesselClassIndex = 0;
$showVesselClassSelect = True;
$showVerifiedSelect = True;
echo '<thead><tr>';
foreach ($header as $value){
	echo "<th>$value</th>";
	array_push($indeces, $value);
	if (strcmp($value, 'verified') == 0){
		$verifiedOccurs = True;
		$verifiedIndex = $index;
	}
	if (strcmp($value, 'id') == 0){
		$idOccurs = True;
		$idIndex = $index;
	}
	if (strcmp($value, 'vessel') == 0){
		$vesselOccurs = True;
		$vesselIndex = $index;
	}
	if (strcmp($value, 'sequence_id') == 0){
		$sequenceIdOccurs = True;
		$sequenceIdIndex = $index;
	}
	if (strcmp($value, 'zone') == 0){
		$zoneOccurs = True;
		$zoneIndex = $index;
	}
	if (strcmp($value, 'vesselClass') == 0){
		$vesselClassOccurs = True;
		$vesselClassIndex = $index;
	}
	$index++;
}
echo '</tr><thead>';

echo '<tbody>';
foreach ($data as $row)
{
	$thisRowsID = '';
	$thisRowsZone = '';
	$thisRowsSequenceId = '';
	if ($idOccurs or $zoneOccurs or $sequenceIdOccurs){
		$i = 0;
		foreach($row as $value){
			if ($i == $idIndex){
				$thisRowsID = $value;
			}
			if ($i == $zoneIndex){
				$thisRowsZone = $value;
			}
			if ($i == $sequenceIdIndex){
				$thisRowsSequenceId = $value;
			}
			$i++;
		}
	}

	$index = 0;
	$id='';
	echo '<tr id='.$thisRowsID.'>';
    foreach($row as $value) {
		
		if ($verifiedOccurs and $index == $verifiedIndex and $showVerifiedSelect){
			$True = ''; $False = ''; $Manual = '';
			if (strcmp($value, 'True') == 0){
				$True = ' selected';
			}
			if (strcmp($value, 'False') == 0){
				$False = ' selected';
			}
			if (strcmp($value, 'Manual') == 0){
				$Manual = ' selected';
			}
			
			echo '<td id="'.$thisRowsID.'" class="'.$indeces[$index].'">
			<select class="verifiedSelector">
			  <option value="None">None</option>
			  <option value="False"'.$False.'>False</option>
			  <option value="True"'.$True.'>True</option>
			  <option value="Manual"'.$Manual.'>Manual</option>
			</select></td>';
			//echo "<td>$value Yes</td>";
		}elseif ($vesselOccurs and $index == $vesselIndex and $zoneOccurs and $sequenceIdOccurs){
			echo "<td class='$indeces[$index]' data-url='../../vesselAnalysis/".$thisRowsSequenceId."/pipeline1/results/".$thisRowsZone."/".$value.".png'><button type='button' class='btn btn-info btn-lg' data-toggle='modal' data-target='#myModal'>$value</button></td>";
		}elseif ($vesselClassOccurs and $index == $vesselClassIndex and $showVesselClassSelect){
			$artery = ''; $vein = ''; 
			if ($value == 1){
				$artery = ' selected';
			}
			if ($value == 2){
				$vein = ' selected';
			}
			echo '<td id="'.$thisRowsID.'" class="'.$indeces[$index].'">
			<select class="vesselClassSelector">
			  <option value="0">None</option>
			  <option value="1"'.$artery.'>Artery</option>
			  <option value="2"'.$vein.'>Vein</option>
			</select></td>';
		}
		else{
			echo "<td class='$indeces[$index]'>$value</td>";
		}
		$index++;
	}
	echo '</tr>';
}
echo '</tbody>';
?>