import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ManualIQ — Asistente Tecnico Industrial",
  description:
    "Consulta manuales de equipos industriales en espanol con citas exactas.",
  manifest: "/manifest.json",
  themeColor: "#1d4ed8",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
