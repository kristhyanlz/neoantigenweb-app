import { useEffect, useState } from 'react';
import { Container, Box, InputLabel, FormControl, Select, MenuItem, TextField, Button, FormLabel, Autocomplete } from '@mui/material';

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
  const [inputType, setInputType] = useState('Peptide');

  useEffect(() => {
    console.log(inputType);
  }, [inputType]);

  return (
    <Container maxWidth='xs' sx={styles.container}>
      <Box sx={styles.titulo}> 
          SUBMISSION
          <BiotechIcon fontSize='large' />
      </Box>

      <Box component='form' onSubmit={null} sx={styles.formulario}>

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
        />

        <Autocomplete
          fullWidth
          disablePortal
          options={hlaData}
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
      </Box>
    </Container>
  )
}