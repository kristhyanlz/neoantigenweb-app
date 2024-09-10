import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Container, Box, InputLabel, FormControl, Select, MenuItem, TextField, Button, FormLabel, Autocomplete } from '@mui/material';

//TOAST
import { toast } from 'react-toastify';

import {FileUpload as FileUploadIcon, Biotech as BiotechIcon} from '@mui/icons-material';

import {hlaData} from '../data/unique_hla';

const styles = {
  container: {
    flex: 1,
    justifyContent: 'flex-start',
    marginTop: 2,
    paddingTop: 0,
    paddingHorizontal: 0,
  },
  titulo:{
    textAlign: 'center',
    fontSize: 32,
    paddingTop: 0,
    paddingBottom: 3,
    fontFamily: 'Candara',
  },
  formEle: {
    paddingHorizontal: 10,
    paddingTop: 0,
  },
  formulario:{
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    mt: 1,
    minWidth: 20
  },
  pb:{
    paddingBottom: 3
  }
};

const inputTypes = [
  {key: 'Peptide', value: 'Peptide'},
  {key: 'Fasta', value: 'Fasta'},
]


export default function Home() {
  const navigate = useNavigate();
  
  const [inputType, setInputType] = useState('Peptide');
  const [inputPeptide, setInputPeptide] = useState('');
  const [inputMhc, setInputMhc] = useState('');
  const [inputHla, setInputHla] = useState('');

  useEffect(() => {
    console.log(`inputType: ${inputType}`);
  }, [inputType]);

  //MHC + peptide
  const submition = async (event) => {
    event.preventDefault()
    let jsonQuery = []
    const allPeptides = inputPeptide.split("\n")
    for (let i = 0; i < allPeptides.length; i++) {
      if (inputType === 'Fasta' && allPeptides[i].startsWith('>')) {
        continue
      }
      
      jsonQuery.push({
        'hla': inputHla,
        'peptide': allPeptides[i],
        'mhc': inputMhc,
      })
    }

    console.log(jsonQuery)

    const submitQuery = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(jsonQuery)
    })

    if (!submitQuery.ok) {
      if (submitQuery.status === 500) {
        toast.error("Revisa los datos enviados")
      } else {
        toast.error("Error en la solicitud")
      }
    } else {
      const predictdata = await submitQuery.json()
      toast.success("Solicitud exitosa")
      console.log(predictdata)

      localStorage.setItem('predictdata', JSON.stringify(predictdata))
      navigate('/Output')
    }
  }

  return (
    <Container maxWidth='xs' sx={styles.container}>
      <Box sx={styles.titulo}> 
          SUBMISSION
          <BiotechIcon fontSize='large' />
      </Box>

      <Box component='form' onSubmit={submition} sx={styles.formulario}>

        <FormControl fullWidth sx={styles.pb}>
          <InputLabel id="lbl-input-type">Input type</InputLabel>
          <Select
            labelId="lbl-input-type"
            label="Input type"
            name='inputType'
            onChange={(val) => setInputType(val.target.value)}
            required
            defaultValue={inputTypes[0].value}
          >
            {
              inputTypes.map((ele) => 
                <MenuItem key={ele.key} value={ele.value}>
                  {ele.value}
                </MenuItem>
              )
            }
          </Select>
        </FormControl>

        <TextField
          fullWidth
          label={`Paste one or more sequences in ${inputType} format`}
          multiline
          rows={4}
          sx={styles.pb}
          onChange={(val) => {
            setInputPeptide(val.target.value)
          }}
        />

        <Autocomplete
          fullWidth
          disablePortal
          options={hlaData}
          onChange={(event, newValue) => {
            setInputMhc(newValue.id);
            setInputHla(newValue.label);
          }}
          renderInput={(Params) => (
            <TextField
              {...Params}
              label="Select HLA alleles"
              required
            />
          )}
        />

        <FormControl fullWidth sx={styles.pb}>
          <FormLabel> {`Or upload a file in ${inputType} format`} </FormLabel>
          <Button endIcon={<FileUploadIcon />}>
            File TXT
          </Button>
          <Button endIcon={<FileUploadIcon />}>
            File CSV
          </Button>
        </FormControl>

        <Button type='submit' variant='contained' color='primary'>
          Submit
        </Button>
      </Box>
    </Container>
  )
}