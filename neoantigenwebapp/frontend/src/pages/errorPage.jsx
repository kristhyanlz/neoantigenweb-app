import { Box } from "@mui/material";

export default function ErrorPage() {
  //const error = useRouteError();
  //console.error(error);

  return (
    <Box id="error-page">
      <h1>
        ¡Vaya!
      </h1>
      <p>
        Lo sentimos, la dirección no se encuentra disponible.
      </p>
    </Box>
  );
}