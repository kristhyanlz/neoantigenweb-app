import MUIDataTable from "mui-datatables";
import { useState, useEffect } from "react";

export default function Output() {

  const columns = [
    {
      name: "hla",
      label: "HLA",
      options: {
        filter: true,
        sort: false,
      }
    },
    {
      name: "mhc",
      label: "MHC",
      options: {
      filter: true,
      sort: true,
      }
    },
    {
      name: "peptide",
      label: "Peptide",
      options: {
      filter: true,
      sort: false,
      }
    },
  {
    name: "prediction",
    label: "Prediction",
    options: {
    filter: true,
    sort: false,
    }
  },
  {
    name: "score",
    label: "Score",
    options: {
    filter: true,
    sort: false,
    }
  },
  ];

  const [data, setData] = useState([]);

  useEffect(() => {
    if (localStorage.getItem('predictdata') != ''){
      setData( JSON.parse(localStorage.getItem('predictdata')) )
    }
    //localStorage.setItem('predictdata', '')
  }, []);

  const options = {
    filterType: 'checkbox',
  };

  return (
    <MUIDataTable
      title={"Predictions"}
      data={data}
      columns={columns}
      options={options}
    />
  )
}