import { Route, Routes } from "react-router-dom";

import AppShell from "@/components/AppShell";
import Home from "@/pages/Home";
import Analyze from "@/pages/Analyze";
import Method from "@/pages/Method";
import Privacy from "@/pages/Privacy";
import NotFound from "@/pages/NotFound";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Home />} />
        <Route path="analyze" element={<Analyze />} />
        <Route path="method" element={<Method />} />
        <Route path="privacy" element={<Privacy />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
