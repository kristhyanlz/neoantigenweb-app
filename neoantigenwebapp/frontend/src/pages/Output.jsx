import MUIDataTable from "mui-datatables";

export default function Output() {

  const columns = [
  {
    name: "name",
    label: "MHC",
    options: {
    filter: true,
    sort: true,
    }
  },
  {
    name: "company",
    label: "Peptide",
    options: {
    filter: true,
    sort: false,
    }
  },
  {
    name: "city",
    label: "Core",
    options: {
    filter: true,
    sort: false,
    }
  },
  {
    name: "state",
    label: "% Bind level",
    options: {
    filter: true,
    sort: false,
    }
  },
  ];

  const data = [
  { name: "HLA-A*01:01", company: "CTGACCATGT", city: "CTGACCATGT", state: "87.857" },
  { name: "HLA-A*01:01", company: "TGCCGCTTAC", city: "TGCCGCTTAC", state: "45.784" },
  { name: "HLA-A*01:01", company: "GATGAGGAGT", city: "GATGAGGAGT", state: "89.451" },
  { name: "HLA-A*01:01", company: "CCCTCAGGGT", city: "CCCTCAGGGT", state: "90.123" },
  ];

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