import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Container, Box, FormControl, TextField, Button, FormLabel, Autocomplete } from '@mui/material';

//TOAST
import { toast } from 'react-toastify';

import {FileUpload as FileUploadIcon, Biotech as BiotechIcon, ContentPasteGo as ContentPasteGoIcon} from '@mui/icons-material';

import {hlaData} from '../data/unique_hla';
import sampleData from '../data/sample_data';

import VisuallyHiddenInput from '../components/VisuallyHiddenInput';

import styles from '../components/submissionStyles';

export default function Home() {
  const navigate = useNavigate();
  
  const [inputPeptide, setInputPeptide] = useState('');
  const [inputMhc, setInputMhc] = useState('');
  const [inputHla, setInputHla] = useState('');

  //MHC + peptide
  const submition = async (event) => {
    event.preventDefault()
    let jsonQuery = []
    const allPeptides = inputPeptide.split("\n")
    for (let i = 0; i < allPeptides.length; i++) {
      if (allPeptides[i].startsWith('>')){
        continue
      }

      jsonQuery.push({
        'hla': inputHla,
        'peptide': allPeptides[i],
        'mhc': inputMhc,
      })
    }

    console.log(jsonQuery)

    try {
      const submitQuery = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(jsonQuery)
      })

      const predictdata = await submitQuery.json()
      toast.success("Success!")
      console.log(predictdata)

      localStorage.setItem('predictdata', JSON.stringify(predictdata))
      navigate('/Output')
      
    } catch (error) {
        toast.error("Error in the request")
    }
  }

  const handleFileChange = (event) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type === "text/plain") {
        const reader = new FileReader();
        reader.onload = (e) => {
          console.log(e.target.result); // Imprime el contenido del archivo en la consola
          setInputPeptide(e.target.result);
        };
        reader.readAsText(file); // Leer archivo como texto
      } else {
        toast.error("You must select a .txt file");
      }
    }
  };

  const loadSampleData = (event) => {
    event.preventDefault()
    setInputPeptide(sampleData.join('\n'))
    setInputHla('HLA-A*01:01')
  }


  return (
    <Container maxWidth='xs' sx={styles.container}>
      <Box sx={styles.titulo}> 
          SUBMISSION
          <BiotechIcon fontSize='large' />
      </Box>

      <Box component='form' onSubmit={submition} sx={styles.formulario}>
        <TextField
          fullWidth
          multiline
          label='Paste in peptide or fasta format. One per line'
          rows={4}
          sx={styles.pb}
          value={inputPeptide}
          onChange={(val) => {
            setInputPeptide(val.target.value)
          }}
        />

        <Autocomplete
          fullWidth
          disablePortal
          options={hlaData}
          value={inputHla}
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
          <FormLabel> {`Or upload a file`} </FormLabel>
          <Button
            component="label"
            endIcon={<FileUploadIcon />}
          >
            File TXT
            <VisuallyHiddenInput
              type='file'
              accept='.txt'
              onChange={handleFileChange}
            />
          </Button>
        </FormControl>

        <FormControl fullWidth sx={styles.sampleButton}>
          <Button
            endIcon={<ContentPasteGoIcon/>}
            onClick={loadSampleData}
          >
            Load sample data
          </Button>
        </FormControl>

        <Button type='submit' variant='contained' color='primary'>
          Submit
        </Button>
      </Box>
    </Container>
  )
}